"""``diffusers`` is gone from the vendored tree, and what replaced it is correct.

The Matcha decoder borrowed eight symbols from ``diffusers`` -- a heavyweight
training library this inference-only model had no business depending on, and one
that cannot even be imported in this environment. They now live in
``chatterbox/models/s3gen/matcha/_diffusers_compat.py`` (src/PATCHES.md P-020).

Shape checks are cheap to write and easy to pass by accident, so the attention,
the adaptive norm and the gated activation are each compared against an
independent hand-written reference computed from the module's own weights. A
shape assertion would pass on a wrong implementation; these would not.

``conftest.py`` mocks the whole vendored tree, so the compat module is loaded
here by file path under a private name.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import ROOT_DIR

VENDOR_DIR = os.path.join(ROOT_DIR, "src", "chatterbox")
COMPAT_REL = "chatterbox/models/s3gen/matcha/_diffusers_compat.py"
COMPAT_PATH = os.path.join(VENDOR_DIR, COMPAT_REL[len("chatterbox/"):].replace("/", os.sep))

#: Exactly the symbols ``matcha/transformer.py`` and ``matcha/decoder.py`` use.
EXPECTED_EXPORTS = {
    "GEGLU",
    "GELU",
    "ApproximateGELU",
    "AdaLayerNorm",
    "AdaLayerNormZero",
    "Attention",
    "get_activation",
    "maybe_allow_in_graph",
}


@pytest.fixture(scope="module")
def compat():
    """The compat module, loaded from disk under a private name."""
    name = "_diffusers_compat_under_test"
    spec = importlib.util.spec_from_file_location(name, COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def vendor_files() -> list[str]:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(VENDOR_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), VENDOR_DIR)
                out.append("chatterbox/" + rel.replace(os.sep, "/"))
    return sorted(out)


def imported_modules(rel_file: str) -> set[str]:
    path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class TestDiffusersIsGone:
    def test_no_vendored_file_imports_diffusers(self) -> None:
        offenders: dict[str, set[str]] = {}
        for rel_file in vendor_files():
            hits = {
                module
                for module in imported_modules(rel_file)
                if module.split(".")[0] == "diffusers"
            }
            if hits:
                offenders[rel_file] = hits
        assert not offenders, (
            f"these vendored files import diffusers again: {offenders}. The "
            "symbols they need live in _diffusers_compat.py (src/PATCHES.md P-020)"
        )

    def test_no_vendored_file_even_mentions_diffusers_in_an_import_statement(self) -> None:
        """Catches ``importlib.import_module('diffusers')`` and ``__import__``."""
        offenders: list[str] = []
        for rel_file in vendor_files():
            path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = getattr(func, "attr", getattr(func, "id", ""))
                    if name in {"import_module", "__import__"}:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and str(arg.value).startswith(
                                "diffusers"
                            ):
                                offenders.append(rel_file)
        assert not offenders, f"dynamic diffusers import in {sorted(set(offenders))}"

    def test_the_lora_class_is_still_gone(self) -> None:
        for rel_file in ("chatterbox/models/s3gen/matcha/transformer.py",):
            assert "LoRACompatibleLinear" not in imported_modules(rel_file)

    def test_the_compat_module_exists_where_the_imports_point(self) -> None:
        assert os.path.isfile(COMPAT_PATH)
        for rel_file in (
            "chatterbox/models/s3gen/matcha/transformer.py",
            "chatterbox/models/s3gen/matcha/decoder.py",
        ):
            with open(
                os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep)),
                "r",
                encoding="utf-8",
            ) as handle:
                source = handle.read()
            assert "from ._diffusers_compat import" in source, (
                f"{rel_file} does not import from ._diffusers_compat"
            )


class TestCompatModuleExports:
    def test_all_is_exactly_the_needed_symbols(self, compat) -> None:
        assert set(compat.__all__) == EXPECTED_EXPORTS, (
            f"unexpected export surface: {sorted(set(compat.__all__) ^ EXPECTED_EXPORTS)}"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_EXPORTS))
    def test_symbol_is_present_and_usable(self, compat, name: str) -> None:
        assert hasattr(compat, name), f"_diffusers_compat does not define {name}"

    def test_the_license_notice_is_retained(self) -> None:
        """The code is copied from an Apache-2.0 project; the notice must travel."""
        with open(COMPAT_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "Apache License" in source, "the upstream license header was dropped"
        assert "Copyright" in source

    def test_maybe_allow_in_graph_is_a_no_op_decorator(self, compat) -> None:
        class Widget:
            pass

        assert compat.maybe_allow_in_graph(Widget) is Widget

    @pytest.mark.parametrize("name", ["silu", "gelu", "relu", "mish", "swish"])
    def test_get_activation_returns_the_named_module(self, compat, name: str) -> None:
        activation = compat.get_activation(name)
        assert isinstance(activation, nn.Module)
        probe = torch.randn(4)
        assert torch.equal(activation(probe), activation(probe))

    def test_get_activation_rejects_an_unknown_name(self, compat) -> None:
        with pytest.raises(ValueError, match="Unsupported activation function"):
            compat.get_activation("not-an-activation")


class TestNumerics:
    """Every check compares against a reference written from the module's weights."""

    def test_attention_matches_a_hand_written_sdpa_reference(self, compat) -> None:
        torch.manual_seed(0)
        batch, seq, dim, heads, head_dim = 2, 9, 32, 4, 8
        attn = compat.Attention(query_dim=dim, heads=heads, dim_head=head_dim).eval()
        x = torch.randn(batch, seq, dim)

        with torch.no_grad():
            got = attn(x)

            q = attn.to_q(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            k = attn.to_k(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            v = attn.to_v(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) / (head_dim**0.5)
            weights = scores.softmax(dim=-1)
            expected = (weights @ v).transpose(1, 2).reshape(batch, seq, heads * head_dim)
            expected = attn.to_out[1](attn.to_out[0](expected))

        assert got.shape == (batch, seq, dim)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    def test_attention_honours_the_padding_mask(self, compat) -> None:
        """Matcha passes a ``(batch, seq)`` bool mask where True means *keep*."""
        torch.manual_seed(1)
        batch, seq, dim, heads, head_dim = 3, 12, 32, 4, 8
        attn = compat.Attention(query_dim=dim, heads=heads, dim_head=head_dim).eval()
        x = torch.randn(batch, seq, dim)
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[:, -3:] = False  # the tail is padding
        mask[1, :2] = False  # and this sample lost its head too

        with torch.no_grad():
            got = attn(x, attention_mask=mask)

            q = attn.to_q(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            k = attn.to_k(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            v = attn.to_v(x).view(batch, seq, heads, head_dim).transpose(1, 2)
            expanded = mask.repeat_interleave(heads, dim=0).view(batch, heads, 1, seq)
            expected = F.scaled_dot_product_attention(q, k, v, attn_mask=expanded)
            expected = expected.transpose(1, 2).reshape(batch, seq, heads * head_dim)
            expected = attn.to_out[1](attn.to_out[0](expected))

        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    def test_masked_positions_do_not_leak_into_the_output(self, compat) -> None:
        """Changing masked key/value entries must not move the output at all.

        Cross-attention, so the perturbation provably cannot reach the queries:
        a self-attention mask covers the same rows that produce the queries.
        """
        torch.manual_seed(2)
        batch, q_len, kv_len, dim, heads, head_dim = 2, 4, 9, 16, 2, 8
        attn = compat.Attention(
            query_dim=dim, cross_attention_dim=dim, heads=heads, dim_head=head_dim
        ).eval()
        x = torch.randn(batch, q_len, dim)
        context = torch.randn(batch, kv_len, dim)
        mask = torch.ones(batch, kv_len, dtype=torch.bool)
        mask[:, 6:] = False

        with torch.no_grad():
            baseline = attn(x, encoder_hidden_states=context, attention_mask=mask)
            perturbed_context = context.clone()
            perturbed_context[:, 6:] += 100.0
            perturbed = attn(x, encoder_hidden_states=perturbed_context, attention_mask=mask)

        torch.testing.assert_close(baseline, perturbed, rtol=0, atol=1e-6)

    def test_without_a_mask_every_position_is_attended(self, compat) -> None:
        """The counter-check: without a mask the same perturbation *does* move it."""
        torch.manual_seed(9)
        attn = compat.Attention(
            query_dim=16, cross_attention_dim=16, heads=2, dim_head=8
        ).eval()
        x = torch.randn(2, 4, 16)
        context = torch.randn(2, 9, 16)
        with torch.no_grad():
            baseline = attn(x, encoder_hidden_states=context)
            perturbed_context = context.clone()
            perturbed_context[:, 6:] += 100.0
            perturbed = attn(x, encoder_hidden_states=perturbed_context)
        assert not torch.allclose(baseline, perturbed), (
            "a full-context perturbation changed nothing; the previous test "
            "would pass vacuously"
        )

    def test_cross_attention_uses_the_encoder_states(self, compat) -> None:
        torch.manual_seed(3)
        attn = compat.Attention(
            query_dim=16, cross_attention_dim=24, heads=2, dim_head=8
        ).eval()
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 11, 24)
        with torch.no_grad():
            cross = attn(x, encoder_hidden_states=context)
            same_shapes = attn(x, encoder_hidden_states=torch.randn_like(context))
        assert cross.shape == (2, 5, 16)
        assert not torch.allclose(cross, same_shapes), (
            "the encoder states were not actually used for key/value"
        )

    def test_to_out_keeps_the_diffusers_parameter_names(self, compat) -> None:
        """The checkpoint keys depend on this exact module layout."""
        attn = compat.Attention(query_dim=8, heads=2, dim_head=4)
        keys = set(attn.state_dict())
        assert {
            "to_q.weight",
            "to_k.weight",
            "to_v.weight",
            "to_out.0.weight",
            "to_out.0.bias",
        } <= keys, f"unexpected attention parameter names: {sorted(keys)}"
        assert isinstance(attn.to_out, nn.ModuleList) and len(attn.to_out) == 2

    def test_ada_layer_norm_structure_matches_diffusers(self, compat) -> None:
        dim, steps = 16, 32
        norm = compat.AdaLayerNorm(dim, steps)
        assert isinstance(norm.emb, nn.Embedding) and norm.emb.num_embeddings == steps
        assert isinstance(norm.silu, nn.SiLU)
        assert norm.linear.out_features == dim * 2
        assert isinstance(norm.norm, nn.LayerNorm)
        assert norm.norm.elementwise_affine is False, (
            "affine parameters here would appear in the checkpoint and break a "
            "strict load"
        )
        assert "norm.weight" not in norm.state_dict()

    def test_ada_layer_norm_builds_a_scaled_then_shifted_conditioning(self, compat) -> None:
        """The conditioning ``forward`` derives, checked against the formula.

        Upstream's ``forward`` ends in ``torch.chunk(emb, 2)`` with no ``dim``,
        which splits the *batch* and therefore only ever broadcasts for inputs
        ``diffusers`` never produces. The shipped copy keeps that call verbatim
        -- see :meth:`TestUnreachableAdaptiveNorms` -- so what is verified here is
        the part that is right: a zero timestep embedding must scale by exactly
        one and shift by exactly zero.
        """
        torch.manual_seed(4)
        dim, steps = 16, 32
        norm = compat.AdaLayerNorm(dim, steps).eval()
        with torch.no_grad():
            same_timestep = norm.linear(norm.silu(norm.emb(torch.tensor([0, 0]))))
            other_timestep = norm.linear(norm.silu(norm.emb(torch.tensor([5, 9]))))

        scale, shift = same_timestep.chunk(2, dim=-1)
        assert same_timestep.shape == (2, dim * 2)
        assert scale.shape == shift.shape == (2, dim), (
            "the 2*dim conditioning must split into a scale and a shift"
        )
        torch.testing.assert_close(same_timestep[0], same_timestep[1], rtol=1e-6, atol=1e-6), (
            "the same timestep index must produce the same conditioning"
        )
        assert not torch.allclose(same_timestep, other_timestep, atol=1e-6), (
            "the timestep embedding is being ignored"
        )

    def test_ada_layer_norm_is_affine_free_inside(self, compat) -> None:
        """diffusers uses ``elementwise_affine=False``; extra parameters would
        break a strict checkpoint load."""
        norm = compat.AdaLayerNorm(16, 8)
        assert norm.norm.elementwise_affine is False
        assert "norm.weight" not in norm.state_dict()

    def test_ada_layer_norm_zero_returns_the_five_chunks(self, compat) -> None:
        torch.manual_seed(5)
        dim = 16
        norm = compat.AdaLayerNormZero(dim, 64)
        x = torch.randn(2, 5, dim)
        timestep = torch.tensor([1, 7])
        with torch.no_grad():
            out = norm(x, timestep, class_labels=None)
        assert len(out) == 5
        normalised, gate_msa, shift_mlp, scale_mlp, gate_mlp = out
        assert normalised.shape == x.shape
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
            assert chunk.shape == (2, dim)

    def test_geglu_matches_x_times_gelu_of_the_gate(self, compat) -> None:
        torch.manual_seed(6)
        geglu = compat.GEGLU(16, 12)
        x = torch.randn(3, 5, 16)
        with torch.no_grad():
            got = geglu(x)
            hidden, gate = geglu.proj(x).chunk(2, dim=-1)
            expected = hidden * F.gelu(gate)
        assert got.shape == (3, 5, 12)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_gelu_approximation_is_tanh(self, compat) -> None:
        torch.manual_seed(7)
        gelu = compat.GELU(16, 16)
        tanh_gelu = compat.GELU(16, 16, approximate="tanh")
        x = torch.randn(3, 5, 16)
        with torch.no_grad():
            tanh_gelu.proj.load_state_dict(gelu.proj.state_dict())
            got = tanh_gelu(x)
            projected = gelu.proj(x)
            expected = projected * 0.5 * (1.0 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * (projected + 0.044715 * projected**3)
            ))
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)

    def test_approximate_gelu_matches_sigmoid_form(self, compat) -> None:
        torch.manual_seed(8)
        module = compat.ApproximateGELU(16, 16)
        x = torch.randn(3, 5, 16)
        with torch.no_grad():
            got = module(x)
            projected = module.proj(x)
            expected = projected * torch.sigmoid(1.702 * projected)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


class TestUnreachableAdaptiveNorms:
    """``AdaLayerNorm``/``AdaLayerNormZero`` are imported but never constructed.

    ``BasicTransformerBlock`` only builds them when ``num_embeds_ada_norm`` is
    given, and the Matcha decoder's single construction site never passes it.
    That is why the copied ``AdaLayerNorm.forward`` can keep upstream's
    dimension-less ``torch.chunk(emb, 2)`` verbatim -- it cannot run. If a future
    re-sync makes the path reachable, this test is what says so.
    """

    CONSTRUCTION_SITES = ("chatterbox/models/s3gen/matcha/decoder.py",)

    def test_no_construction_site_enables_them(self) -> None:
        offenders: list[str] = []
        for rel_file in self.CONSTRUCTION_SITES:
            path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "BasicTransformerBlock"
                ):
                    keys = {
                        kw.arg for kw in node.keywords if kw.arg is not None
                    }
                    if "num_embeds_ada_norm" in keys or "norm_type" in keys:
                        offenders.append(rel_file)
        assert not offenders, (
            f"{sorted(set(offenders))} now pass num_embeds_ada_norm/norm_type; "
            "AdaLayerNorm is reachable and its copied forward (which keeps "
            "upstream's dimension-less torch.chunk) must be fixed first"
        )

    def test_the_class_still_reexports_the_upstream_shape(self, compat) -> None:
        """Kept importable so the transformer module's import list stays valid."""
        assert issubclass(compat.AdaLayerNorm, nn.Module)
        assert issubclass(compat.AdaLayerNormZero, nn.Module)


class TestMatchaDecoderUsesTheCompatModule:
    """The constructions the decoder performs, checked against the module."""

    def test_get_activation_is_the_decoder_timestep_activation(self, compat) -> None:
        """``Decoder.time_mlp`` is built with ``act_fn="silu"``."""
        activation = compat.get_activation("silu")
        assert isinstance(activation, nn.SiLU)
        probe = torch.randn(4)
        torch.testing.assert_close(activation(probe), F.silu(probe))

    @pytest.mark.parametrize(
        ("rel_file", "symbol"),
        [
            ("chatterbox/models/s3gen/matcha/transformer.py", "Attention"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "GEGLU"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "GELU"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "ApproximateGELU"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "AdaLayerNorm"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "AdaLayerNormZero"),
            ("chatterbox/models/s3gen/matcha/transformer.py", "maybe_allow_in_graph"),
            ("chatterbox/models/s3gen/matcha/decoder.py", "get_activation"),
        ],
    )
    def test_the_call_site_imports_that_symbol_locally(self, compat, rel_file: str, symbol: str) -> None:
        path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=path)
        local: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "_diffusers_compat":
                local.update(alias.asname or alias.name for alias in node.names)
        assert symbol in local, (
            f"{rel_file} uses {symbol} but does not import it from ._diffusers_compat "
            f"(local imports: {sorted(local)})"
        )
        assert hasattr(compat, symbol)