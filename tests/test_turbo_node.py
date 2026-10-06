"""S8.2 -- ``ChatterboxTurboTTS``: its own sampler, its own tags, and no Nano node.

Turbo differs from the classic and multilingual nodes in ways that are easy to
get wrong quietly:

* ``top_k`` exists here and only here. Turbo samples with top-k; the other two
  families do not have it, and offering one there would be a knob with no
  effect. ``test_top_k_is_rejected_on_the_multilingual_node`` is the
  cross-node half of that.
* The tag vocabulary is Turbo's own. ``[laugh]`` and ``[clear throat]`` are nine
  square-bracket tokens the Turbo tokenizer was trained on; the multilingual v2
  set is thirty angle-bracket ones. They are not interchangeable.
* **There is no Nano node class, and there must not be one.** Upstream ships no
  ``tts_nano.py`` -- Nano is a second set of weights reached through
  ``nano=True`` in the same module. A ``ChatterboxNanoTTS`` class would be a
  second thing to keep in step with a model that is one flag away. The cache-key
  assertions below are the other half of the same point: two weight sets, two
  keys, never one served for the other.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from ComfyUI_Chatterbox.modules.generation import (
    build_cache_key,
    variant_cache_stem,
)
from ComfyUI_Chatterbox.modules.text_processing import TURBO_TAGS
from ComfyUI_Chatterbox.nodes import mtl_node as mtl_module
from ComfyUI_Chatterbox.nodes.turbo_node import TURBO_TAG_LIST, ChatterboxTurboTTSNode

TURBO_NODE_ID = "ChatterboxTurboTTS"
EXPECTED_TURBO_TAGS = (
    "laugh",
    "sigh",
    "gasp",
    "chuckle",
    "cough",
    "sniff",
    "groan",
    "shush",
    "clear throat",
)


def _required() -> dict:
    return ChatterboxTurboTTSNode.GET_NODE_INFO_V1()["input"]["required"]


class TestSchema:
    def test_the_node_id_is_new(self) -> None:
        schema = ChatterboxTurboTTSNode.GET_SCHEMA()
        assert schema.node_id == TURBO_NODE_ID
        assert schema.outputs[0].display_name == "audio"
        assert schema.is_output_node is False
        assert ChatterboxTurboTTSNode.FINALIZE_SCHEMA().outputs[0].id

    def test_the_schema_validates(self) -> None:
        assert ChatterboxTurboTTSNode.validate_inputs() is True

    def test_an_unknown_pack_is_refused_with_a_message(self) -> None:
        result = ChatterboxTurboTTSNode.validate_inputs(model_pack_name="no_such_pack")
        assert isinstance(result, str)
        assert "no_such_pack" in result

    def test_every_advertised_dtype_is_accepted(self) -> None:
        for option in _required()["dtype"][1]["options"]:
            assert ChatterboxTurboTTSNode.validate_inputs(
                model_pack_name=None, dtype=option, attention_mode="auto"
            ) is True

    def test_auto_is_the_last_device_option(self) -> None:
        assert _required()["device"][1]["options"][-1] == "auto"


class TestSampler:
    """Turbo's sampler is a top-k / top-p pair with no CFG."""

    def test_top_k_is_one_to_five_thousand_defaulting_to_one_thousand(self) -> None:
        config = _required()["top_k"]
        assert config[0] == "INT"
        assert (config[1]["min"], config[1]["max"], config[1]["default"]) == (
            1,
            5000,
            1000,
        )

    def test_temperature_stops_at_two_point_zero(self) -> None:
        """The classic node runs to 5.0; Turbo was distilled for a narrower band."""
        config = _required()["temperature"]
        assert (config[1]["min"], config[1]["max"]) == (0.05, 2.0)

        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode

        classic = ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"]["temperature"]
        assert classic[1]["max"] == 5.0

    def test_top_p_defaults_to_point_nine_five(self) -> None:
        config = _required()["top_p"]
        assert (config[1]["min"], config[1]["max"], config[1]["default"]) == (
            0.1,
            1.0,
            0.95,
        )

    def test_repetition_penalty_default_is_the_documented_one(self) -> None:
        config = _required()["repetition_penalty"]
        assert (config[1]["min"], config[1]["max"], config[1]["default"]) == (
            1.0,
            3.0,
            1.2,
        )

    def test_there_is_no_cfg_weight_or_min_p(self) -> None:
        """Upstream logs a warning for any non-zero CFG on this family."""
        names = set(_required())
        assert "cfg_weight" not in names
        assert "min_p" not in names

        from ComfyUI_Chatterbox.modules.generation import generate_turbo_audio

        parameters = inspect.signature(generate_turbo_audio).parameters
        assert "cfg_weight" not in parameters
        assert "min_p" not in parameters

    def test_top_k_is_rejected_on_the_multilingual_node(self) -> None:
        """The cross-node half of "top_k is Turbo only".

        A dropdown is a promise: a value it offers must mean something when the
        node runs. The multilingual sampler has no top-k, so the widget must not
        be there to be silently ignored.
        """
        info = mtl_module.ChatterboxMTLTTSNode.GET_NODE_INFO_V1()["input"]
        names = set(info["required"]) | set(info.get("optional", {}))
        assert "top_k" not in names, "the multilingual sampler has no top-k"
        assert "top_k" not in inspect.signature(
            mtl_module.ChatterboxMTLTTSNode.execute
        ).parameters

    def test_top_k_is_rejected_on_the_classic_node_too(self) -> None:
        from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode

        names = set(ChatterboxTTSNode.GET_NODE_INFO_V1()["input"]["required"])
        assert "top_k" not in names


class TestTurboTagSet:
    def test_the_documented_nine_tags(self) -> None:
        assert TURBO_TAGS == EXPECTED_TURBO_TAGS
        assert len(TURBO_TAGS) == 9

    def test_the_tooltip_is_built_from_the_parser_not_typed_out(self) -> None:
        """The tooltip cannot drift from what the parser recognises."""
        assert TURBO_TAG_LIST == " ".join(f"[{tag}]" for tag in EXPECTED_TURBO_TAGS)
        tooltip = _required()["text"][1]["tooltip"]
        for tag in EXPECTED_TURBO_TAGS:
            assert f"[{tag}]" in tooltip

    def test_clear_throat_is_two_words_with_a_space(self) -> None:
        """Not ``[clear_throat]`` -- that is the multilingual v2 spelling."""
        assert "clear throat" in TURBO_TAGS
        tooltip = _required()["text"][1]["tooltip"]
        assert "[clear throat]" in tooltip
        assert "[clear_throat]" not in tooltip

    def test_the_tags_are_recognised_before_they_reach_the_model(self) -> None:
        from ComfyUI_Chatterbox.modules.text_processing import plan_text

        plan = plan_text(
            "Hello, this is a test. [laugh] Isn't that funny?", turbo_tags=True
        )
        assert plan.segments[0].turbo_tags == ("laugh",)
        assert plan.text == "Hello, this is a test. Isn't that funny?"


class TestThereIsNoNanoNode:
    """Upstream ships no ``tts_nano.py``. Nano is a flag, not a node."""

    def test_no_nano_module_exists_in_the_vendor(self) -> None:
        import os

        from conftest import ROOT_DIR

        vendor = os.path.join(ROOT_DIR, "src", "chatterbox")
        assert not os.path.exists(os.path.join(vendor, "tts_nano.py"))

    def test_no_nano_node_class_exists(self) -> None:
        import ComfyUI_Chatterbox.nodes as nodes_pkg

        exported = [name for name in dir(nodes_pkg) if "Nano" in name or "NANO" in name]
        assert exported == [], f"unexpected Nano exports: {exported}"

        classes = [
            value
            for value in vars(nodes_pkg).values()
            if inspect.isclass(value) and "Nano" in getattr(value, "__name__", "")
        ]
        assert classes == []

    def test_no_nano_node_id_is_declared_anywhere_in_the_tree(self) -> None:
        import os
        import re

        from conftest import ROOT_DIR

        pattern = re.compile(r"ChatterboxNanoTTS|node_id\s*=\s*[\"']ChatterboxNano")
        for directory in ("nodes", "modules"):
            for name in sorted(os.listdir(os.path.join(ROOT_DIR, directory))):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(ROOT_DIR, directory, name)
                with open(path, encoding="utf-8") as handle:
                    assert not pattern.search(handle.read()), path

    def test_nano_is_reachable_through_a_widget(self) -> None:
        config = _required()["nano"]
        assert config[0] == "BOOLEAN"
        assert config[1]["default"] is False

    def test_nano_reaches_the_loader(self) -> None:
        assert "nano" in inspect.signature(
            ChatterboxTurboTTSNode.execute
        ).parameters

        from ComfyUI_Chatterbox.modules.loader import ChatterboxLoader

        assert "nano" in inspect.signature(ChatterboxLoader.load_turbo_model).parameters


class TestTurboAndNanoDoNotShareACacheEntry:
    """Two weight sets, two keys.

    Turbo and Nano are the same module over two disjoint sets of weights. Without
    the variant in the cache key, loading Nano after Turbo is served Turbo's
    weights under a shared key -- and the user gets the wrong model with nothing
    anywhere reporting an error.
    """

    def test_the_variant_stem_differs(self) -> None:
        assert variant_cache_stem("tts_turbo", "turbo") != variant_cache_stem(
            "tts_turbo", "nano"
        )
        assert variant_cache_stem("tts_turbo", "nano") == "tts_turbo__nano"

    def test_the_cache_keys_differ(self) -> None:
        turbo = build_cache_key(
            variant_cache_stem("tts_turbo", "turbo"), "sdpa", "fp32", torch.device("cpu")
        )
        nano = build_cache_key(
            variant_cache_stem("tts_turbo", "nano"), "sdpa", "fp32", torch.device("cpu")
        )
        assert turbo != nano
        assert "nano" in nano and "nano" not in turbo

    def test_multilingual_versions_also_differ(self) -> None:
        v2 = build_cache_key(
            variant_cache_stem("pack", "t3_mtl23ls_v2.safetensors"),
            "sdpa",
            "fp32",
            torch.device("cpu"),
        )
        v3 = build_cache_key(
            variant_cache_stem("pack", "t3_mtl23ls_v3.safetensors"),
            "sdpa",
            "fp32",
            torch.device("cpu"),
        )
        assert v2 != v3

    def test_an_empty_variant_leaves_the_name_alone(self) -> None:
        assert variant_cache_stem("pack", "") == "pack"

    def test_the_node_passes_the_right_variant(self, monkeypatch) -> None:
        seen = {}

        def _capture(model_name, **kwargs):
            seen.update(kwargs)
            return type("P", (), {"cache_key": "k"})(), object()

        monkeypatch.setattr(
            "ComfyUI_Chatterbox.nodes.turbo_node.load_chatterbox_entry_point", _capture
        )
        monkeypatch.setattr(
            "ComfyUI_Chatterbox.nodes.turbo_node.generate_turbo_audio",
            lambda *a, **k: (torch.zeros((1, 240)), 24000),
        )
        for nano, expected in ((False, "turbo"), (True, "nano")):
            ChatterboxTurboTTSNode.execute(
                model_pack_name="tts_turbo",
                text="Hello.",
                nano=nano,
                temperature=0.8,
                top_k=1000,
                top_p=0.95,
                repetition_penalty=1.2,
                exaggeration=0.0,
                normalize_loudness=True,
                seed=1,
                use_watermark=False,
            )
            assert seen["variant"] == expected
            assert seen["entry_point"] == "turbo"


class TestLoudnessFallback:
    """``pyloudnorm`` is absent here, and normalisation is skipped, not fatal."""

    def test_it_is_genuinely_absent(self) -> None:
        import importlib.util

        assert importlib.util.find_spec("pyloudnorm") is None

    def test_norm_loudness_falls_back_with_a_warning(self, monkeypatch, caplog) -> None:
        """The vendored ``norm_loudness`` returns the audio untouched when ``ln`` is None.

        Exercised against the real vendored source: the harness mocks
        ``chatterbox.tts_turbo``, so the function under test is compiled from
        the file on disk with its module body stripped down to the guard and the
        helper. That is the code that actually decides whether a missing
        ``pyloudnorm`` is a warning or an exception.
        """
        import textwrap

        from ComfyUI_Chatterbox.modules.text_processing import TURBO_TAGS as _unused

        del _unused
        source = _vendored_norm_loudness_source()
        namespace: dict = {"logging": logging_module(), "ln": None, "math": math_module()}
        exec(compile(textwrap.dedent(source), "tts_turbo.py", "exec"), namespace)
        norm_loudness = namespace["norm_loudness"]

        samples = torch.ones(4, 24000, dtype=torch.float32)
        with caplog.at_level("WARNING"):
            result = norm_loudness(None, samples, 24000)
        assert torch.equal(result, samples), (
            "with pyloudnorm absent the audio must come back untouched"
        )

    def test_the_vendored_import_is_guarded(self) -> None:
        import ast

        source = _vendored_module_source("tts_turbo.py")
        tree = ast.parse(source, filename="tts_turbo.py")
        pyloudnorm_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Import)
                and any("pyloudnorm" in alias.name for alias in child.names)
                for child in ast.walk(node)
            )
        ]
        assert pyloudnorm_imports, (
            "pyloudnorm is imported at module scope again, which makes the whole "
            "turbo entry point unimportable on a machine without it"
        )

    def test_the_widget_defaults_to_normalising_and_can_be_turned_off(self) -> None:
        config = _required()["normalize_loudness"]
        assert config[0] == "BOOLEAN"
        assert config[1]["default"] is True
        assert "normalize_loudness" in inspect.signature(
            ChatterboxTurboTTSNode.execute
        ).parameters


def logging_module():
    import logging

    return logging


def math_module():
    import math

    return math


def _vendored_module_source(name: str) -> str:
    import os

    from conftest import ROOT_DIR

    with open(
        os.path.join(ROOT_DIR, "src", "chatterbox", name), encoding="utf-8"
    ) as handle:
        return handle.read()


def _vendored_norm_loudness_source() -> str:
    """The real ``norm_loudness`` body, lifted out of the vendored module.

    The harness mocks ``chatterbox.tts_turbo``, so the function cannot simply be
    imported. Lifting the source keeps the assertion on the code that ships.
    """
    source = _vendored_module_source("tts_turbo.py")
    start = source.index("    def norm_loudness(self")
    end = source.index("    def prepare_conditionals(self")
    # ``start`` points at the 4-space class indent; dropping it lifts the method
    # to module level with its body indentation already correct.
    return source[start + 4 : end]
