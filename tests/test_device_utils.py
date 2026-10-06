"""S5.3 -- ``modules/device_utils.py``.

These run against the **real** ``comfy.model_management``: the node is a ComfyUI
node, and a fake would only test the fake. The properties pinned are the ones
that silently broke saved workflows:

* an option list whose indices shift, because ComfyUI binds widget values by
  index -- ``"auto"`` is appended last, never inserted;
* a device string that appears in the dropdown but does not resolve, which
  ComfyUI accepts and torch then rejects inside the model constructor.
"""

from __future__ import annotations

import ast
import logging

import pytest
import torch

import comfy.model_management as mm

from modules import device_utils
from modules.device_utils import (
    DEVICE_AUTO,
    DEVICE_CPU,
    get_device_display_name,
    get_device_options,
    get_offload_device,
    get_torch_device,
    is_gpu_device,
    should_use_bf16,
    should_use_fp16,
)


class TestGetDeviceOptions:
    def test_auto_is_the_last_option(self) -> None:
        """Not merely present: LAST. Inserting it shifts every saved index."""
        options = get_device_options()

        assert options[-1] == DEVICE_AUTO, (
            f"'auto' must be last so existing widget indices stay valid; got {options}"
        )

    def test_auto_appears_exactly_once(self) -> None:
        assert get_device_options().count(DEVICE_AUTO) == 1

    def test_cpu_is_offered(self) -> None:
        assert DEVICE_CPU in get_device_options()

    def test_options_are_unique(self) -> None:
        options = get_device_options()

        assert len(options) == len(set(options))

    def test_every_option_resolves_to_a_torch_device(self) -> None:
        for option in get_device_options():
            device = get_torch_device(option)
            assert isinstance(device, torch.device), f"{option!r} did not resolve"

    def test_every_comfyui_gpu_option_resolves(self) -> None:
        """Whatever ComfyUI itself offers has to survive ``get_torch_device``."""
        for option in mm.get_gpu_device_options():
            device = get_torch_device(option)
            assert isinstance(device, torch.device), f"{option!r} did not resolve"


class TestAutoResolution:
    def test_auto_uses_comfyui_default_device(self) -> None:
        assert get_torch_device(DEVICE_AUTO) == mm.get_torch_device()

    def test_none_is_treated_as_auto(self) -> None:
        assert get_torch_device(None) == mm.get_torch_device()

    def test_auto_follows_the_machine(self, monkeypatch) -> None:
        """On a GPU box it is the GPU; on a CPU-only box it is ``cpu``."""
        monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
        assert get_torch_device(DEVICE_AUTO).type == "cpu"

        monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cuda"))
        assert get_torch_device(DEVICE_AUTO).type == "cuda"


class TestExplicitDevices:
    def test_cpu_is_honoured(self) -> None:
        assert get_torch_device(DEVICE_CPU) == torch.device("cpu")

    def test_cuda_alias_maps_to_the_comfyui_default(self) -> None:
        assert get_torch_device("cuda") == mm.get_torch_device()

    def test_cuda_with_index_maps_to_the_comfyui_default(self) -> None:
        assert get_torch_device("cuda:0") == mm.get_torch_device()

    def test_unknown_device_falls_back_with_a_warning(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            device = get_torch_device("quantum-gpu")

        assert device == mm.get_torch_device()
        assert any("quantum-gpu" in record.getMessage() for record in caplog.records)

    def test_unavailable_mps_falls_back(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(torch.backends, "mps", None, raising=False)
        with caplog.at_level(logging.WARNING):
            device = get_torch_device("mps")

        assert device == mm.get_torch_device()
        assert any("mps" in record.getMessage() for record in caplog.records)

    def test_unavailable_xpu_falls_back(self, monkeypatch) -> None:
        class _Xpu:
            @staticmethod
            def is_available() -> bool:
                return False

        monkeypatch.setattr(torch, "xpu", _Xpu(), raising=False)

        assert get_torch_device("xpu") == mm.get_torch_device()


class TestOffloadDevice:
    def test_matches_comfyui_intermediate_device(self) -> None:
        assert get_offload_device() == mm.intermediate_device()


class TestPrecisionHelpers:
    def test_should_use_fp16_defaults_to_the_comfyui_device(self) -> None:
        assert should_use_fp16() == mm.should_use_fp16(mm.get_torch_device())

    def test_should_use_bf16_defaults_to_the_comfyui_device(self) -> None:
        assert should_use_bf16() == mm.should_use_bf16(mm.get_torch_device())

    def test_explicit_device_is_honoured(self) -> None:
        assert should_use_fp16(torch.device("cpu")) == mm.should_use_fp16(torch.device("cpu"))

    def test_cpu_is_not_a_gpu(self) -> None:
        assert is_gpu_device(DEVICE_CPU) is False

    def test_cuda_is_a_gpu(self) -> None:
        assert is_gpu_device("cuda") is True

    def test_non_string_input_is_not_a_gpu(self) -> None:
        assert is_gpu_device(None) is False


class TestDisplayNames:
    def test_every_option_has_a_non_empty_label(self) -> None:
        for option in get_device_options():
            label = get_device_display_name(option)
            assert isinstance(label, str) and label, f"no label for {option!r}"

    def test_cpu_label(self) -> None:
        assert get_device_display_name(DEVICE_CPU) == "CPU"

    def test_labels_are_unique(self) -> None:
        labels = [get_device_display_name(option) for option in get_device_options()]

        assert len(labels) == len(set(labels))


class TestModelManagementHelper:
    def test_returns_the_real_module(self) -> None:
        assert device_utils._get_model_management() is mm

    def test_result_is_cached(self) -> None:
        assert device_utils._get_model_management() is device_utils._get_model_management()

    def test_no_comfy_import_at_module_scope(self) -> None:
        with open(device_utils.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=device_utils.__file__)

        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top_level.add(node.module.split(".")[0])

        assert "comfy" not in top_level
        assert "comfy_api" not in top_level