"""S5.4 -- ``modules/dtype_utils.py``.

Three things here are regressions, not conveniences:

* **transformers v5 renamed the config attribute.** ``config.torch_dtype`` is a
  deprecated property on the installed 5.3.0 that warns on every get *and* set.
  ``set_config_dtype`` has to write ``dtype`` instead, and must emit no warning
  while doing so.
* **An explicit dtype is validated, not obeyed.** ``resolve_dtype("bf16")`` on
  hardware that cannot run bf16 raises, instead of handing back a
  ``torch.bfloat16`` the caller discovers three minutes into a CUDA kernel.
* **Quant-resident storage is never cast.** Those bytes are a payload, not
  numbers.

No network, no real weights: the "models" below are a few ``torch.nn.Module``
trees of a few hundred bytes.
"""

from __future__ import annotations

import ast
import logging
import warnings

import pytest
import torch

import comfy.model_management as mm

from modules import dtype_utils
from modules.dtype_utils import (
    DTYPE_AUTO,
    DTYPE_BF16,
    DTYPE_FP16,
    DTYPE_FP32,
    cast_model_to_dtype,
    cast_model_to_dtype_if_needed,
    get_config_dtype,
    get_dtype_options,
    get_dtype_str,
    representative_dtype,
    resolve_dtype,
    set_config_dtype,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class TransformersV5Config:
    """A config shaped like transformers >= 5.0: ``torch_dtype`` is the
    deprecated property alias that warns on every access."""

    def __init__(self, dtype=None) -> None:
        # transformers v5 stores the value on ``dtype``; ``torch_dtype`` is only
        # the deprecated alias that reads and writes it.
        self.dtype = dtype

    @property
    def torch_dtype(self):
        warnings.warn(
            "`torch_dtype` is deprecated; use `dtype` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dtype

    @torch_dtype.setter
    def torch_dtype(self, value):
        warnings.warn(
            "`torch_dtype` is deprecated; use `dtype` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.dtype = value


class LegacyConfig:
    """A transformers 4.x-shaped config: ``torch_dtype`` is a plain attribute."""

    def __init__(self, dtype=None) -> None:
        self.torch_dtype = dtype


class QuantLinear(torch.nn.Module):
    """Stands in for a GGUF / int8 / FP8 linear: packed storage, not numbers."""

    _quant_resident = True

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(8, dtype=torch.uint8))
        self.weight_scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))


class TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32, with_quant: bool = False) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, dtype=dtype)
        self.head = torch.nn.Linear(4, 4, dtype=dtype)
        if with_quant:
            self.q = QuantLinear()


# ---------------------------------------------------------------------------
# Option list
# ---------------------------------------------------------------------------


class TestDtypeOptions:
    def test_auto_leads_the_list(self) -> None:
        assert get_dtype_options()[0] == DTYPE_AUTO

    def test_all_four_options_present(self) -> None:
        assert get_dtype_options() == [DTYPE_AUTO, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32]

    def test_options_resolve(self) -> None:
        device = mm.get_torch_device()
        for option in get_dtype_options():
            assert isinstance(resolve_dtype(option, device), torch.dtype)

    def test_get_dtype_str_round_trips(self) -> None:
        device = mm.get_torch_device()
        for option in ("bf16", "fp16", "fp32"):
            assert get_dtype_str(resolve_dtype(option, device)) == option

    def test_get_dtype_str_rejects_an_unknown_dtype(self) -> None:
        with pytest.raises(ValueError, match="Unknown torch.dtype"):
            get_dtype_str(torch.float64)


# ---------------------------------------------------------------------------
# transformers v5 config attribute
# ---------------------------------------------------------------------------


class TestSetConfigDtype:
    def test_writes_dtype_on_a_v5_config(self) -> None:
        config = TransformersV5Config()

        set_config_dtype(config, torch.bfloat16)

        assert config.dtype is torch.bfloat16

    def test_emits_no_warning_on_a_v5_config(self) -> None:
        """The whole point: the deprecated property is never touched."""
        config = TransformersV5Config()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            set_config_dtype(config, torch.bfloat16)

        assert caught == [], f"set_config_dtype warned: {[str(w.message) for w in caught]}"

    def test_get_config_dtype_on_a_v5_config_does_not_warn(self) -> None:
        config = TransformersV5Config(dtype=torch.float16)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = get_config_dtype(config)

        assert value is torch.float16
        assert caught == []

    def test_writes_torch_dtype_on_a_legacy_config(self) -> None:
        config = LegacyConfig()

        set_config_dtype(config, torch.float32)

        assert config.torch_dtype is torch.float32

    def test_reads_a_legacy_config(self) -> None:
        assert get_config_dtype(LegacyConfig(dtype=torch.bfloat16)) is torch.bfloat16

    def test_string_dtypes_are_resolved_through_torch(self) -> None:
        assert get_config_dtype(LegacyConfig(dtype="float16")) is torch.float16

    def test_a_config_without_a_dtype_reads_as_none(self) -> None:
        assert get_config_dtype(LegacyConfig()) is None


# ---------------------------------------------------------------------------
# resolve_dtype
# ---------------------------------------------------------------------------


class TestResolveDtypeAuto:
    def test_auto_follows_bf16_when_comfyui_says_so(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: True)

        assert resolve_dtype(DTYPE_AUTO, torch.device("cuda")) is torch.bfloat16

    def test_auto_falls_to_fp16(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)
        monkeypatch.setattr(dtype_utils, "should_use_fp16", lambda device=None: True)

        assert resolve_dtype(DTYPE_AUTO, torch.device("cuda")) is torch.float16

    def test_auto_falls_to_fp32(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)
        monkeypatch.setattr(dtype_utils, "should_use_fp16", lambda device=None: False)

        assert resolve_dtype(DTYPE_AUTO, torch.device("cuda")) is torch.float32

    def test_none_behaves_like_auto(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: True)

        assert resolve_dtype(None, torch.device("cuda")) is torch.bfloat16

    def test_auto_without_a_device_uses_the_comfyui_default(self, monkeypatch) -> None:
        seen = {}

        def _remember(device=None):
            seen["device"] = device
            return False

        monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cuda"))
        monkeypatch.setattr(dtype_utils, "should_use_bf16", _remember)
        monkeypatch.setattr(dtype_utils, "should_use_fp16", _remember)

        resolve_dtype(DTYPE_AUTO)

        assert seen["device"] == torch.device("cuda")

    def test_auto_matches_comfyui_on_this_machine(self) -> None:
        device = mm.get_torch_device()
        expected = (
            torch.bfloat16
            if mm.should_use_bf16(device)
            else torch.float16
            if mm.should_use_fp16(device)
            else torch.float32
        )

        assert resolve_dtype(DTYPE_AUTO, device) is expected


class TestResolveDtypeValidation:
    def test_explicit_bf16_on_unsupported_hardware_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)

        with pytest.raises(ValueError, match="bf16"):
            resolve_dtype(DTYPE_BF16, torch.device("cuda"))

    def test_the_rejection_names_the_device_and_suggests_auto(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)

        with pytest.raises(ValueError) as excinfo:
            resolve_dtype(DTYPE_BF16, torch.device("cuda"))

        message = str(excinfo.value)
        assert "cuda" in message
        assert "auto" in message

    def test_explicit_bf16_is_allowed_where_comfyui_supports_it(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: True)

        assert resolve_dtype(DTYPE_BF16, torch.device("cuda")) is torch.bfloat16

    def test_cpu_is_exempt_from_bf16_validation(self, monkeypatch) -> None:
        """torch implements bf16 on CPU; a user asking for it there gets it."""
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)

        assert resolve_dtype(DTYPE_BF16, torch.device("cpu")) is torch.bfloat16

    def test_fp32_is_never_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)
        monkeypatch.setattr(dtype_utils, "should_use_fp16", lambda device=None: False)

        assert resolve_dtype(DTYPE_FP32, torch.device("cuda")) is torch.float32

    def test_unknown_dtype_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dtype"):
            resolve_dtype("int8", torch.device("cpu"))

    def test_validation_error_is_a_value_error_not_a_crashed_kernel(self, monkeypatch) -> None:
        monkeypatch.setattr(dtype_utils, "should_use_bf16", lambda device=None: False)

        try:
            resolve_dtype(DTYPE_BF16, torch.device("cuda"))
        except ValueError as exc:
            assert "cannot" in str(exc)
        else:  # pragma: no cover - the assertion above is the failure path
            pytest.fail("an unsupported bf16 request was silently accepted")


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------


class TestCastModel:
    def test_floating_parameters_are_cast(self) -> None:
        model = TinyModel(dtype=torch.float32)

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.lin.weight.dtype is torch.bfloat16
        assert model.head.weight.dtype is torch.bfloat16

    def test_integer_parameters_are_untouched(self) -> None:
        model = TinyModel()
        model.register_buffer("counter", torch.zeros(3, dtype=torch.int32))

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.counter.dtype is torch.int32

    def test_quant_resident_parameters_are_never_cast(self) -> None:
        model = TinyModel(with_quant=True)
        original_bytes = model.q.weight.clone()

        cast_model_to_dtype(model, torch.bfloat16)

        assert model.q.weight.dtype is torch.uint8, "packed quant bytes were re-typed"
        assert torch.equal(model.q.weight, original_bytes)
        assert model.q.weight_scale.dtype is torch.float32, "the fp32 scale was cast"

    def test_quant_resident_storage_is_never_cast_by_the_if_needed_variant(self) -> None:
        model = TinyModel(dtype=torch.bfloat16, with_quant=True)
        original_bytes = model.q.weight.clone()

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.q.weight.dtype is torch.uint8
        assert torch.equal(model.q.weight, original_bytes)

    def test_cast_to_none_is_a_no_op(self) -> None:
        model = TinyModel(dtype=torch.float32)

        cast_model_to_dtype(model, None)

        assert model.lin.weight.dtype is torch.float32

    def test_if_needed_skips_a_fully_matching_model(self, caplog) -> None:
        model = TinyModel(dtype=torch.bfloat16)

        with caplog.at_level(logging.DEBUG):
            cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.lin.weight.dtype is torch.bfloat16
        assert not [r for r in caplog.records if "Model cast" in r.getMessage()]

    def test_if_needed_casts_a_mismatching_model(self) -> None:
        model = TinyModel(dtype=torch.float32)

        cast_model_to_dtype_if_needed(model, torch.bfloat16)

        assert model.head.weight.dtype is torch.bfloat16

    def test_if_needed_to_none_is_a_no_op(self) -> None:
        model = TinyModel(dtype=torch.float32)

        cast_model_to_dtype_if_needed(model, None)

        assert model.lin.weight.dtype is torch.float32


class TestRepresentativeDtype:
    def test_reads_the_floating_parameter_dtype(self) -> None:
        assert representative_dtype(TinyModel(dtype=torch.float16)) is torch.float16

    def test_ignores_quant_resident_parameters(self) -> None:
        model = TinyModel(dtype=torch.float32, with_quant=True)

        assert representative_dtype(model) is torch.float32

    def test_falls_back_to_the_model_dtype_attribute(self) -> None:
        class NoParams(torch.nn.Module):
            dtype = torch.bfloat16

        assert representative_dtype(NoParams()) is torch.bfloat16

    def test_returns_none_when_there_is_nothing_to_read(self) -> None:
        class Empty(torch.nn.Module):
            pass

        assert representative_dtype(Empty()) is None


class TestNoComfyImportsAtModuleScope:
    def test_module_scope_imports_no_comfy(self) -> None:
        with open(dtype_utils.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=dtype_utils.__file__)

        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level.add(node.module.split(".")[0])

        assert "comfy" not in top_level
        assert "comfy_api" not in top_level

    def test_model_management_helper_is_shared_with_device_utils(self) -> None:
        from modules import device_utils

        assert dtype_utils._get_model_management is device_utils._get_model_management
        assert dtype_utils._get_model_management() is mm

class _PlainPack:
    """Mirrors the vendored pack shape: a *plain* class holding nn.Modules.

    ``src.chatterbox.tts.ChatterboxTTS`` (and the VC/MTL/Turbo packs) are not
    ``nn.Module`` -- their weights hang off ordinary attributes. The dtype
    walkers are handed these objects from ``ChatterboxPatcher.patch_model``,
    so the AttributeError regression (``'ChatterboxTTS' object has no
    attribute 'named_modules'``) is reproduced with this shape, not with a
    ``TinyModel``.
    """

    def __init__(self) -> None:
        self.t3 = torch.nn.Linear(4, 4)
        self.ve = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
        self.conds = {"cond": torch.zeros(2)}  # plain tensor holder: not a param
        # A lying ``dtype`` property: the pre-fix fallback path answered from
        # this when ``named_parameters`` blew up. The walkers must find the
        # real parameter dtype without consulting it.
        self._declared = torch.float16

    @property
    def dtype(self):
        return self._declared


class TestPlainPackContainers:
    def test_cast_reaches_a_plain_pack(self) -> None:
        pack = _PlainPack()
        dtype_utils.cast_model_to_dtype(pack, torch.float16)
        assert pack.t3.weight.dtype is torch.float16
        assert pack.ve[0].weight.dtype is torch.float16
        assert pack.ve[1].weight.dtype is torch.float16

    def test_representative_dtype_walks_instead_of_trusting_the_property(self) -> None:
        pack = _PlainPack()
        assert dtype_utils.representative_dtype(pack) is torch.float32

    def test_named_parameters_of_a_plain_pack_are_dotted(self) -> None:
        names = {name for name, _ in dtype_utils._iter_named_parameters(_PlainPack())}
        assert "t3.weight" in names and "t3.bias" in names
        assert "ve.0.weight" in names and "ve.1.weight" in names
        assert "conds" not in names

    def test_no_duplicate_params_for_shared_submodules(self) -> None:
        shared = torch.nn.Linear(4, 4)
        pack = _PlainPack()
        pack.s3gen = shared
        pack.t3 = shared  # same object under two names
        cast_names = []
        for name, param in dtype_utils._iter_named_parameters(pack):
            cast_names.append(name)
        assert len([n for n in cast_names if n.startswith(("t3.", "s3gen."))]) == 2

    def test_if_needed_is_a_noop_when_already_cast(self) -> None:
        pack = _PlainPack()
        dtype_utils.cast_model_to_dtype(pack, torch.float16)
        before = pack.t3.weight.data_ptr()
        dtype_utils.cast_model_to_dtype_if_needed(pack, torch.float16)
        assert pack.t3.weight.data_ptr() == before

    def test_quant_resident_module_inside_a_plain_pack_is_protected(self) -> None:
        pack = _PlainPack()
        packed = QuantLinear()
        packed._quant_resident = True
        pack.s3gen = packed
        protected = dtype_utils._quant_protected_names(pack)
        assert "s3gen.weight" in protected
        dtype_utils.cast_model_to_dtype(pack, torch.float16)
        assert packed.weight.dtype is not torch.float16

    def test_the_real_vendor_pack_passes_through_the_cast(self) -> None:
        """The exact shape from the field: wrapper.tts_model is a plain pack."""
        pack = _PlainPack()
        current = dtype_utils.representative_dtype(pack)
        dtype_utils.cast_model_to_dtype(pack, torch.float16)
        assert current is torch.float32
        assert pack.t3.weight.dtype is torch.float16
