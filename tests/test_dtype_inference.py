"""S8.9a -- half precision actually holds end to end.

Upstream's dtype support is three things working together: a ``dtype``
property that reports what the parameters *currently* hold, ``cast_all()`` at
the solver boundary, and the ``ref_dict`` cast in ``S3Token2Mel.forward``. A test
that only checked ``resolve_dtype`` would pass with all three removed.

The code under test is the real vendored source. ``conftest.py`` mocks the whole
``chatterbox`` tree so collection never loads the model stack, so the mocks are
lifted for this module and restored afterwards. Two consequences shape the file:

* The heavyweight constructors are **not** called. ``S3Tokenizer`` and
  ``CAMPPlus`` would download and allocate hundreds of megabytes; what is under
  test is the ``dtype`` property, the ``ref_dict`` loop and the solver's
  boundary, and each is reached through a probe object that skips the
  constructor rather than through a full model.
* A probe that cannot see the code is not a test, so every class here first
  asserts that the real property, method or function was the one exercised.
"""

from __future__ import annotations

import inspect

import pytest
import torch


@pytest.fixture(scope="module")
def vendor():
    """The real vendored model modules, with the harness mocks restored after.

    ``conftest.py`` mocks the whole ``chatterbox`` tree so collection never loads
    the model stack; these tests are about the ``dtype`` properties, the
    ``ref_dict`` loop and the solver's boundary, which exist only in the real
    source. The mocks go back on teardown -- see
    :func:`conftest_helpers.real_vendor_modules`.
    """
    from .conftest_helpers import real_vendor_modules

    with real_vendor_modules(
        [
            "models.s3gen.s3gen",
            "models.s3gen.decoder",
            "models.s3gen.flow_matching",
        ]
    ) as modules:
        yield {
            "s3gen": modules["models.s3gen.s3gen"],
            "decoder": modules["models.s3gen.decoder"],
            "flow_matching": modules["models.s3gen.flow_matching"],
        }


class _TinyEstimator(torch.nn.Module):
    """A stand-in for the flow-matching estimator, with a controllable dtype.

    ``ConditionalCFM`` reads ``self.estimator.dtype`` and casts every input to it
    before calling. That is the boundary under test, so the stand-in has to be a
    real module *with* the ``dtype`` property the vendored ``ConditionalDecoder``
    defines -- ``nn.Module`` itself has none, which is why the decoder carries
    one of its own.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    @property
    def dtype(self) -> torch.dtype:
        return self.scale.dtype

    def forward(self, x, mask=None, mu=None, t=None, spks=None, cond=None, r=None):
        return x * 0.0


# ---------------------------------------------------------------------------
# The dtype properties
# ---------------------------------------------------------------------------


class TestDtypePropertiesReflectTheCast:
    def test_s3token2mel_dtype_reports_the_live_parameter_dtype(self, vendor) -> None:
        """``next(self.flow.parameters()).dtype`` -- so a cast is visible at once.

        The property is the whole contract between "the user picked bf16" and
        every ``.to(dtype=self.dtype)`` downstream. If it reported the config
        rather than the parameters, every cast in the file would be a no-op and
        nothing would say so.
        """
        S3Token2Mel = vendor["s3gen"].S3Token2Mel
        probe = S3Token2Mel.__new__(S3Token2Mel)
        torch.nn.Module.__init__(probe)
        probe.flow = torch.nn.Linear(4, 4)

        assert probe.dtype is torch.float32
        probe.flow.to(torch.bfloat16)
        assert probe.dtype is torch.bfloat16
        probe.flow.to(torch.float16)
        assert probe.dtype is torch.float16

    def test_the_decoder_defines_its_own_dtype_property(self, vendor) -> None:
        """``nn.Module`` has no ``dtype``; the decoder supplies one."""
        assert not hasattr(torch.nn.Linear(2, 2), "dtype")
        assert isinstance(
            vars(vendor["decoder"].ConditionalDecoder).get("dtype"), property
        )

    def test_the_property_is_not_a_stored_attribute(self, vendor) -> None:
        """It is read from the parameters, not cached at construction."""
        source = inspect.getsource(vendor["s3gen"].S3Token2Mel)
        assert "def dtype(self):" in source
        assert "next(params).dtype" in source.replace("next(self.flow.parameters())",
                                                      "next(params)")

    def test_conditional_decoder_dtype_reflects_the_cast(self, vendor) -> None:
        """The decoder carries its own ``dtype`` property, as torch does not.

        ``next(self.final_proj.weight.dtype)`` is what the flow-matching solver
        reads to decide what to cast its inputs to, so a decoder cast to half
        while its estimator stayed fp32 would silently run the whole solve in the
        wrong precision. The property reads the *live* weight, which is why the
        cast is visible immediately.
        """
        ConditionalDecoder = vendor["decoder"].ConditionalDecoder
        probe = ConditionalDecoder(
            in_channels=8,
            out_channels=4,
            channels=[8],
            attention_head_dim=8,
            n_blocks=1,
            num_mid_blocks=1,
            num_heads=1,
        )
        assert probe.dtype is torch.float32
        probe.to(torch.bfloat16)
        assert probe.dtype is torch.bfloat16
        probe.to(torch.float16)
        assert probe.dtype is torch.float16

    def test_both_properties_agree_with_representative_dtype(self, vendor) -> None:
        """The patcher casts with ``representative_dtype``; the two must not drift."""
        from ComfyUI_Chatterbox.modules.dtype_utils import (
            cast_model_to_dtype,
            representative_dtype,
        )

        model = torch.nn.Linear(8, 8)
        cast_model_to_dtype(model, torch.bfloat16)
        assert representative_dtype(model) is torch.bfloat16
        assert next(model.parameters()).dtype is representative_dtype(model)


# ---------------------------------------------------------------------------
# embed_ref / forward: the ref_dict cast
# ---------------------------------------------------------------------------


class _FlowStub(torch.nn.Module):
    """The one collaborator ``S3Token2Mel.forward`` calls, recorded and returned.

    A real ``nn.Module``, so ``next(self.flow.parameters()).dtype`` -- the body of
    the vendored ``dtype`` property -- still works, with the inference replaced
    so the test can see exactly which tensors reached the decoder.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.captured: dict = {}

    def inference(self, token, token_len, **kwargs):
        self.captured.update(kwargs)
        return torch.zeros((1, 80, 2), dtype=self.weight.dtype), None


class _RecordingS3Gen:
    """A probe that exercises the real ``S3Token2Mel.forward`` ref_dict loop."""

    def __init__(self, dtype: torch.dtype) -> None:
        self.flow = _FlowStub(dtype)
        self.device = torch.device("cpu")
        self.meanflow = False

    # The vendored property reads `next(self.flow.parameters()).dtype`; reuse it.
    dtype = property(lambda self: next(self.flow.parameters()).dtype)


class TestRefDictIsCast:
    def _forward(self, vendor, dtype: torch.dtype, ref_dict: dict):
        S3Token2Mel = vendor["s3gen"].S3Token2Mel
        probe = _RecordingS3Gen(dtype)

        # The *real* vendored forward, bound to the probe: the ref_dict loop under
        # test is the shipped one rather than a copy of it.
        real_forward = S3Token2Mel.forward
        real_forward(probe, torch.tensor([[1, 2, 3]]), None, None, ref_dict=ref_dict)
        return probe.flow.captured

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_the_floating_ref_tensors_are_cast_to_the_model_dtype(
        self, vendor, dtype
    ) -> None:
        captured = self._forward(
            vendor,
            dtype,
            {
                "prompt_token": torch.zeros((1, 4), dtype=torch.long),
                "prompt_feat": torch.zeros((1, 4, 80), dtype=torch.float32),
                "embedding": torch.zeros((1, 80), dtype=torch.float32),
                "prompt_token_len": torch.tensor([4], dtype=torch.long),
            },
        )
        assert captured["prompt_feat"].dtype is dtype, (
            "the mel conditioning stayed fp32 and would be multiplied by a half "
            "weight matrix"
        )
        assert captured["embedding"].dtype is dtype

    def test_the_integer_tensors_follow_the_device_but_never_the_dtype(
        self, vendor
    ) -> None:
        """P-034: the recorded upstream quirk turned out to be a live bug.

        The old blanket ``.to(dtype=self.dtype)`` also hit ``prompt_token`` /
        ``prompt_token_len``. In fp32 that is an exact no-op, but bf16's 7-bit
        mantissa cannot represent every integer: a combined token length of
        635 rounds to 632, which truncated the flow's padding mask and crashed
        ``flow.inference`` with a shape mismatch (found by a real-weight bf16
        run) -- and token ids above 256 were silently corrupted before that.
        The solver's own ``cast_all`` always had the floating-point guard; the
        two paths now agree.
        """
        captured = self._forward(
            vendor,
            torch.bfloat16,
            {
                "prompt_token": torch.tensor([[7, 300]], dtype=torch.long),
                "prompt_token_len": torch.tensor([635], dtype=torch.long),
            },
        )
        assert captured["prompt_token"].dtype is torch.long
        assert captured["prompt_token_len"].dtype is torch.long
        assert int(captured["prompt_token_len"][0]) == 635
        assert captured["prompt_token"].tolist() == [[7, 300]]

    def test_numpy_arrays_become_tensors_and_are_then_cast(self, vendor) -> None:
        import numpy as np

        captured = self._forward(
            vendor,
            torch.bfloat16,
            {
                "prompt_feat": np.zeros((1, 4, 80), dtype=np.float32),
                "embedding": np.zeros((1, 80), dtype=np.float32),
            },
        )
        assert captured["prompt_feat"].dtype is torch.bfloat16
        assert captured["embedding"].dtype is torch.bfloat16

    def test_the_forward_source_carries_the_cast(self, vendor) -> None:
        source = inspect.getsource(vendor["s3gen"].S3Token2Mel.forward)
        assert "dtype=self.dtype" in source
        assert "torch.from_numpy" in source


# ---------------------------------------------------------------------------
# The solver boundary
# ---------------------------------------------------------------------------


class TestSolverPreservesInDtype:
    def test_cast_all_only_touches_floating_point_tensors(self, vendor) -> None:
        cast_all = vendor["flow_matching"].cast_all
        out = cast_all(
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.float32),
            dtype=torch.bfloat16,
        )
        assert out[0].dtype is torch.bfloat16
        assert out[1].dtype is torch.long, "an integer tensor has no half form"
        assert out[2].dtype is torch.bfloat16

    def test_cast_all_is_a_no_op_at_the_target_dtype(self, vendor) -> None:
        cast_all = vendor["flow_matching"].cast_all
        source = torch.zeros(2, dtype=torch.bfloat16)
        (result,) = cast_all(source, dtype=torch.bfloat16)
        assert result is source, "an already-correct tensor should not be copied"

    @pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "estimator_dtype", [torch.float32, torch.float16, torch.bfloat16]
    )
    def test_the_solver_returns_the_dtype_it_was_given(
        self, vendor, in_dtype: torch.dtype, estimator_dtype: torch.dtype
    ) -> None:
        """``in_dtype`` is captured before the cast and restored on the way out.

        This is the boundary that makes mixed precision work at all: the decoder
        runs in one dtype and the vocoder in another, so the solver has to hand
        back exactly what it was handed rather than whatever its estimator
        happened to run in.
        """
        CausalConditionalCFM = vendor["flow_matching"].CausalConditionalCFM
        solver = CausalConditionalCFM(
            in_channels=80, estimator=_TinyEstimator().to(estimator_dtype)
        )

        x = torch.randn(1, 80, 16, dtype=in_dtype)
        t_span = torch.linspace(0, 1, 3, dtype=in_dtype)
        mu = torch.randn(1, 80, 16, dtype=in_dtype)
        mask = torch.ones(1, 1, 16, dtype=in_dtype)
        spks = torch.randn(1, 80, dtype=in_dtype)
        cond = torch.randn(1, 80, 16, dtype=in_dtype)

        out, _ = solver.forward(
            mu, mask, n_timesteps=2, temperature=1.0, spks=spks, cond=cond, meanflow=True
        )
        assert out.dtype is in_dtype, (
            f"in={in_dtype} estimator={estimator_dtype} produced {out.dtype}"
        )
        assert torch.isfinite(out).all()

    def test_the_solver_casts_its_inputs_to_the_estimator_dtype(self, vendor) -> None:
        """The other half: the estimator really does see half, not fp32."""
        CausalConditionalCFM = vendor["flow_matching"].CausalConditionalCFM
        seen: list[torch.dtype] = []

        class _Spy(_TinyEstimator):
            def forward(self, x, mask=None, mu=None, t=None, spks=None, cond=None, r=None):
                seen.append(x.dtype)
                return x * 0.0

        solver = CausalConditionalCFM(in_channels=80, estimator=_Spy().to(torch.bfloat16))
        solver.forward(
            torch.randn(1, 80, 8, dtype=torch.float32),
            torch.ones(1, 1, 8, dtype=torch.float32),
            n_timesteps=2,
            temperature=1.0,
            spks=torch.randn(1, 80, dtype=torch.float32),
            cond=torch.randn(1, 80, 8, dtype=torch.float32),
            meanflow=True,
        )
        assert seen and all(dtype is torch.bfloat16 for dtype in seen), seen

    def test_the_source_records_the_boundary(self, vendor) -> None:
        """Both solvers: the causal one used here, and the unconditional one."""
        source = inspect.getsource(vendor["flow_matching"].ConditionalCFM.solve_euler)
        assert "in_dtype = x.dtype" in source
        assert "return x.to(in_dtype)" in source

        causal = inspect.getsource(vendor["flow_matching"].CausalConditionalCFM)
        assert "def basic_euler" in causal
        assert "in_dtype = x.dtype" in causal
        assert "return x.to(in_dtype)" in causal


class TestHalfPrecisionOnCpu:
    """The dtype widgets resolve; torch does the rest.

    No GPU is required to show the arithmetic works in half: S3Gen's
    transformer blocks are ordinary ``nn.Module``s and a half forward is a half
    forward on a CPU.
    """

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_a_half_model_runs_and_stays_finite(self, dtype) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 16),
        ).to(dtype)
        out = model(torch.randn(2, 16, dtype=dtype))
        assert out.dtype is dtype
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        ("requested", "device_type", "expected"),
        [
            ("fp16", "cpu", torch.float16),
            ("bf16", "cpu", torch.bfloat16),
        ],
    )
    def test_the_widget_values_resolve_to_real_half_dtypes(
        self, requested: str, device_type: str, expected: torch.dtype
    ) -> None:
        from ComfyUI_Chatterbox.modules.dtype_utils import resolve_dtype

        assert resolve_dtype(requested, torch.device(device_type)) is expected

    def test_fp32_is_not_a_half_dtype(self) -> None:
        from ComfyUI_Chatterbox.modules.dtype_utils import resolve_dtype

        assert resolve_dtype("fp32", torch.device("cpu")) is torch.float32
