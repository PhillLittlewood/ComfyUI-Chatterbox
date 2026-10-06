"""S5.5 -- ``modules/attention_utils.py``.

The rule under test: **a mode string carried by a saved workflow is not a
capability claim.** A graph saved on a box with SageAttention replays verbatim on
one without it; resolution has to notice and fall back rather than let the mode
reach the loader, where it fails as ``RuntimeError("Incompatible hardware/setup
for SageAttention.")`` several minutes into a model load.

Hardware is simulated by monkeypatching ``torch.cuda`` -- no GPU of a
particular architecture is required to run this file.
"""

from __future__ import annotations

import ast
import builtins
import logging
import sys

import pytest
import torch

from modules import attention_utils
from modules.attention_utils import (
    ATTENTION_MODES,
    SAGE_SUPPORTED_ARCHS,
    check_dtype_attention_compatible,
    check_flash_attention_available,
    check_sage_attention_compatible,
    get_attn_implementation_for_load,
    get_available_attention_modes,
    resolve_attention_mode,
    sage_arch_code,
)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Simulate a CUDA device with a chosen compute capability."""

    def _install(available: bool = True, capability=(8, 6)) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)

    return _install


@pytest.fixture
def no_flash_attention(monkeypatch):
    monkeypatch.setattr(
        attention_utils, "check_flash_attention_available", lambda: False
    )


class TestSageArchSupport:
    def test_supported_archs_is_a_frozenset(self) -> None:
        assert isinstance(SAGE_SUPPORTED_ARCHS, frozenset)

    def test_sm80_and_sm90_are_supported(self) -> None:
        assert {"sm80", "sm90"} <= SAGE_SUPPORTED_ARCHS

    def test_blackwell_datacenter_archs_are_rejected(self) -> None:
        """sm100/sm103 have no sage branch and must not fall through to SM90."""
        assert "sm100" not in SAGE_SUPPORTED_ARCHS
        assert "sm103" not in SAGE_SUPPORTED_ARCHS

    def test_sm75_is_rejected(self) -> None:
        assert "sm75" not in SAGE_SUPPORTED_ARCHS

    def test_every_supported_arch_is_spelled_as_sage_spells_it(self) -> None:
        for arch in SAGE_SUPPORTED_ARCHS:
            assert arch.startswith("sm") and arch[2:].isdigit()

    @pytest.mark.parametrize("capability", [(10, 0), (10, 3)])
    def test_blackwell_is_incompatible(self, fake_cuda, capability, caplog) -> None:
        fake_cuda(capability=capability)
        with caplog.at_level(logging.WARNING):
            assert check_sage_attention_compatible() is False
        assert any("not one of the architectures" in r.getMessage() for r in caplog.records)

    def test_ampere_is_compatible(self, fake_cuda) -> None:
        fake_cuda(capability=(8, 6))

        assert check_sage_attention_compatible() is True

    def test_no_cuda_is_incompatible(self, fake_cuda) -> None:
        fake_cuda(available=False)

        assert check_sage_attention_compatible() is False

    def test_missing_library_is_incompatible(self, monkeypatch) -> None:
        monkeypatch.setattr(attention_utils, "SAGE_ATTENTION_AVAILABLE", False)

        assert check_sage_attention_compatible() is False

    def test_arch_code_is_major_times_ten_plus_minor(self, fake_cuda) -> None:
        fake_cuda(capability=(8, 9))

        assert sage_arch_code() == 89

    def test_compatibility_never_raises_without_cuda(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert check_sage_attention_compatible() is False


class TestAvailableAttentionModes:
    def test_eager_and_sdpa_are_always_offered(self) -> None:
        modes = get_available_attention_modes()

        assert "eager" in modes
        assert "sdpa" in modes

    def test_modes_are_unique(self) -> None:
        modes = get_available_attention_modes()

        assert len(modes) == len(set(modes))

    def test_sage_is_omitted_off_hardware(self, fake_cuda, no_flash_attention) -> None:
        fake_cuda(available=False)

        assert "sage" not in get_available_attention_modes()

    def test_sage_is_omitted_on_an_unsupported_arch(
        self, fake_cuda, no_flash_attention
    ) -> None:
        fake_cuda(capability=(10, 0))

        assert "sage" not in get_available_attention_modes()

    def test_sage_is_offered_on_supported_hardware(
        self, fake_cuda, no_flash_attention
    ) -> None:
        fake_cuda(capability=(8, 9))

        modes = get_available_attention_modes()
        assert "sage" in modes or not attention_utils.SAGE_ATTENTION_AVAILABLE

    def test_flash_attention_is_omitted_without_the_package(
        self, fake_cuda, no_flash_attention
    ) -> None:
        fake_cuda()

        assert "flash_attention_2" not in get_available_attention_modes()

    def test_every_offered_mode_is_in_the_declared_mode_list(self) -> None:
        assert set(get_available_attention_modes()) <= set(ATTENTION_MODES)

    def test_every_offered_mode_resolves_to_itself(self, no_flash_attention) -> None:
        for mode in get_available_attention_modes():
            assert resolve_attention_mode(mode) == mode

    def test_offered_modes_survive_the_4bit_swap(self, no_flash_attention) -> None:
        for mode in get_available_attention_modes():
            if mode in ("eager", "flash_attention_2"):
                continue
            assert resolve_attention_mode(mode, quantize_4bit=True) == mode


class TestResolveAttentionMode:
    def test_sage_falls_back_to_sdpa_without_cuda(self, fake_cuda, caplog) -> None:
        fake_cuda(available=False)

        with caplog.at_level(logging.WARNING):
            resolved = resolve_attention_mode("sage")

        assert resolved == "sdpa", (
            "a saved workflow naming 'sage' on a box without sage support must "
            "resolve, not die in the loader"
        )
        assert any("falling back to 'sdpa'" in r.getMessage() for r in caplog.records)

    def test_sage_falls_back_to_sdpa_on_an_unsupported_arch(
        self, fake_cuda, caplog
    ) -> None:
        fake_cuda(capability=(10, 0))

        assert resolve_attention_mode("sage") == "sdpa"

    def test_sage_falls_back_to_sdpa_without_the_library(self, monkeypatch) -> None:
        monkeypatch.setattr(attention_utils, "SAGE_ATTENTION_AVAILABLE", False)

        assert resolve_attention_mode("sage") == "sdpa"

    @pytest.mark.skipif(
        not attention_utils.SAGE_ATTENTION_AVAILABLE,
        reason="sageattention is not installed on this machine",
    )
    def test_sage_survives_on_supported_hardware(self, fake_cuda, monkeypatch) -> None:
        monkeypatch.setattr(attention_utils, "SAGE_ATTENTION_AVAILABLE", True)
        fake_cuda(capability=(8, 9))

        assert resolve_attention_mode("sage") == "sage"

    def test_flash_attention_falls_back_to_sdpa(self, no_flash_attention, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_attention_mode("flash_attention_2")

        assert resolved == "sdpa"
        assert any("flash_attention_2" in r.getMessage() for r in caplog.records)

    def test_four_bit_downgrades_eager_to_sdpa(self, no_flash_attention, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_attention_mode("eager", quantize_4bit=True)

        assert resolved == "sdpa"
        assert any("4-bit" in r.getMessage() for r in caplog.records)

    def test_four_bit_downgrades_flash_to_sdpa(self, no_flash_attention) -> None:
        assert resolve_attention_mode("flash_attention_2", quantize_4bit=True) == "sdpa"

    def test_four_bit_leaves_sdpa_alone(self, no_flash_attention) -> None:
        assert resolve_attention_mode("sdpa", quantize_4bit=True) == "sdpa"

    def test_unknown_mode_falls_back_to_eager(self, no_flash_attention, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            resolved = resolve_attention_mode("flash-attn-9")

        assert resolved == "eager"
        assert any("Unknown attention mode" in r.getMessage() for r in caplog.records)

    def test_every_resolution_is_a_declared_mode(self, fake_cuda) -> None:
        for mode in ATTENTION_MODES + ["", "nonsense", None]:
            assert resolve_attention_mode(mode) in ATTENTION_MODES


class TestAttnImplementationForLoad:
    def test_sage_loads_as_sdpa(self) -> None:
        """transformers has no 'sage' implementation; the patch happens after."""
        assert get_attn_implementation_for_load("sage") == "sdpa"

    @pytest.mark.parametrize("mode", ["eager", "sdpa", "flash_attention_2"])
    def test_other_modes_pass_through(self, mode) -> None:
        assert get_attn_implementation_for_load(mode) == mode

    def test_no_load_implementation_is_sage(self) -> None:
        for mode in ATTENTION_MODES:
            assert get_attn_implementation_for_load(mode) != "sage"


class TestFlashAttentionProbe:
    def test_absent_package_is_unavailable(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "flash_attn":
                raise ImportError("no flash_attn here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)

        assert check_flash_attention_available() is False

    def test_no_cuda_is_unavailable_even_with_the_package(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "flash_attn", object())
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert check_flash_attention_available() is False


class TestDtypeAttentionCrossCheck:
    def test_fp32_with_sage_is_rejected(self) -> None:
        message = check_dtype_attention_compatible("fp32", "sage")

        assert message is not None
        assert "fp32" in message and "sage" in message

    def test_bf16_with_sage_is_fine(self) -> None:
        assert check_dtype_attention_compatible("bf16", "sage") is None

    def test_auto_with_sage_is_fine(self) -> None:
        assert check_dtype_attention_compatible("auto", "sage") is None

    def test_four_bit_exempts_the_check(self) -> None:
        assert check_dtype_attention_compatible("fp32", "sage", True) is None

    def test_no_attention_widget_means_nothing_to_check(self) -> None:
        assert check_dtype_attention_compatible("fp32", None) is None

    @pytest.mark.parametrize("mode", ["eager", "sdpa", "flash_attention_2"])
    def test_other_backends_never_conflict(self, mode) -> None:
        assert check_dtype_attention_compatible("fp32", mode) is None


class TestNoComfyImportsAtModuleScope:
    def test_attention_utils_imports_no_comfy(self) -> None:
        with open(attention_utils.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=attention_utils.__file__)

        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level.add(node.module.split(".")[0])

        assert "comfy" not in top_level
        assert "comfy_api" not in top_level