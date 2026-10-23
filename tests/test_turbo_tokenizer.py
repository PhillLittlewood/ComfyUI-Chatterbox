"""Proof that the 19 upstream Turbo/Nano tags land on the ids upstream gave them.

Upstream ships the paralinguistic vocabulary as *added tokens*: its
``added_tokens.json`` (published by both ``ResembleAI/chatterbox-turbo`` and
``ResembleAI/chatterbox-nano``, byte-identical in the two) maps the 19
bracketed spellings onto the contiguous block of ids 50257-50275, and the
vocabulary the tokenizer reports is 50276 entries long. A tag therefore lands
on its id because the *pack's own tokenizer* carries the added token -- not
because this repository says so.

That is worth proving twice, in two independent ways, because the two halves
fail differently:

* **An always-on code-level proof** over the vendored ``tts_turbo.py``. The
  vendor tree is mocked for the whole suite (``conftest.py``), so these checks
  read the source instead of importing it. They pin the three facts that make
  ids 50257-50275 meaningful -- the model sizes its text vocabulary at 50276,
  builds the tokenizer from the pack directory, and warns when the tokenizer's
  length is not 50276 -- plus the one that makes the ids *reachable*: the
  bracketed tag has to survive the text cleanup and arrive at the tokenizer
  exactly as it was typed. The cleanup is executed from its own source, so
  this is the vendored behaviour, not a restatement of it.

* **An env-gated real-tokenizer proof.** ``RUN_CHATTERBOX_E2E=1`` together with
  ``CHATTERBOX_MODEL_DIR=<pack dir>`` loads the pack's own tokenizer and asks
  it about every tag. With the variables unset the tests skip with a named
  reason, which is the correct outcome on a machine with no pack (and in CI).

The id table below is the upstream truth, transcribed from those two
``added_tokens.json`` files. **This test owns that table**: nothing in the
repository is allowed to move an id, and a change upstream shows up here as a
failure with a diff, not as a silent behaviour change.

**No vendored patch.** The ids are in the tokenizer upstream ships, so the
vendored tree needs no change and ``src/PATCHES.md`` gets no new row. The
proof is that it does not need one.
"""

from __future__ import annotations

import ast
import functools
import json
import os
from typing import Any, Callable

import pytest

from conftest import ROOT_DIR

#: Opt-in switch, following ``tests/test_e2e_smoke.py``.
ENV_ENABLE = "RUN_CHATTERBOX_E2E"
#: Absolute path to a downloaded chatterbox pack directory.
ENV_MODEL_DIR = "CHATTERBOX_MODEL_DIR"

#: The upstream truth: ``added_tokens.json`` as published by
#: ``ResembleAI/chatterbox-turbo`` and ``ResembleAI/chatterbox-nano`` (the two
#: files are byte-identical). Keyed by the exact bracketed spelling upstream
#: uses -- the tokenizer matches added tokens case-sensitively, so the
#: spelling is part of the contract and not a display detail.
UPSTREAM_TAG_IDS: dict[str, int] = {
    "[angry]": 50257,
    "[fear]": 50258,
    "[surprised]": 50259,
    "[whispering]": 50260,
    "[advertisement]": 50261,
    "[dramatic]": 50262,
    "[narration]": 50263,
    "[crying]": 50264,
    "[happy]": 50265,
    "[sarcastic]": 50266,
    "[clear throat]": 50267,
    "[sigh]": 50268,
    "[shush]": 50269,
    "[cough]": 50270,
    "[groan]": 50271,
    "[sniff]": 50272,
    "[gasp]": 50273,
    "[chuckle]": 50274,
    "[laugh]": 50275,
}

#: The text vocabulary the vendored model sizes itself for. The 19 tags are
#: the last ids of it: 50257-50275, with 50276 the count.
TEXT_VOCAB_SIZE = 50276
#: The contiguous block the 19 tags occupy.
TAG_IDS: tuple[int, ...] = tuple(range(50257, 50276))

#: The T3 checkpoint each family ships. The tokenizer is shared between them;
#: the weights are not -- so a pack directory proves a family only when it
#: holds that family's weights.
VARIANTS: tuple[tuple[str, str], ...] = (
    ("turbo", "t3_turbo_v1.safetensors"),
    ("nano", "t3_nano_v1.safetensors"),
)

TURBO_SOURCE = os.path.join(ROOT_DIR, "src", "chatterbox", "tts_turbo.py")


# ---------------------------------------------------------------------------
# Reading the vendored source
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _turbo_tree() -> ast.Module:
    """The vendored Turbo entry point, parsed. Never imported."""
    with open(TURBO_SOURCE, "r", encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=TURBO_SOURCE)


@functools.lru_cache(maxsize=1)
def _punc_norm() -> Callable[[str], str]:
    """The vendored ``punc_norm``, recovered by executing its own source.

    ``conftest.py`` mocks the whole vendor tree, so the function cannot be
    imported here. It is pure string manipulation with no free names, so its
    source is executed in isolation -- the ``tests/test_vendor_signatures.py``
    idiom -- and what comes back is the function the model runs before it
    tokenizes, not a description of it.
    """
    for node in ast.walk(_turbo_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == "punc_norm":
            namespace: dict[str, Any] = {}
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, TURBO_SOURCE, "exec"), namespace)  # noqa: S102
            return namespace["punc_norm"]
    raise AssertionError("punc_norm is no longer defined in tts_turbo.py")


def _keyword_argument(tree: ast.Module, func_name: str, keyword: str) -> Any:
    """The value of ``keyword`` in a call to ``func_name``, or ``None``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == func_name):
            continue
        for keyword_node in node.keywords:
            if keyword_node.arg == keyword and isinstance(keyword_node.value, ast.Constant):
                return keyword_node.value.value
    return None


def _tokenizer_source_argument(tree: ast.Module) -> str | None:
    """The name of the variable the tokenizer is built from, or ``None``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "from_pretrained"):
            continue
        owner = func.value
        if not (isinstance(owner, ast.Name) and owner.id == "AutoTokenizer"):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
    return None


def _tokenizer_length_guard(tree: ast.Module) -> int | None:
    """The number the tokenizer's length is compared against, or ``None``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        if not isinstance(node.ops[0], ast.NotEq):
            continue
        left = node.left
        if not (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Name)
            and left.func.id == "len"
        ):
            continue
        if not left.args or not isinstance(left.args[0], ast.Name):
            continue
        if left.args[0].id != "tokenizer":
            continue
        comparator = node.comparators[0]
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
            return comparator.value
    return None


def _tokenizer_call_argument(tree: ast.Module) -> str | None:
    """The name of the value handed to ``self.tokenizer(...)``, or ``None``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "tokenizer"):
            continue
        owner = func.value
        if not (isinstance(owner, ast.Name) and owner.id == "self"):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            return node.args[0].id
    return None


# ---------------------------------------------------------------------------
# The always-on half: the code that consumes the ids
# ---------------------------------------------------------------------------


class TestTheVendoredCodeExpectsThoseIds:
    """What the vendored model does with the tokenizer, proved from source.

    Each check is a statement about the code the model runs, so a re-sync that
    changed the vocabulary size, the tokenizer source or the text handed to it
    fails here -- with the vendored tree still mocked, and without a pack.
    """

    def test_the_text_vocabulary_is_sized_for_the_tag_ids(self) -> None:
        """``text_tokens_dict_size`` must cover 50257-50275 and one past them."""
        size = _keyword_argument(_turbo_tree(), "T3Config", "text_tokens_dict_size")
        assert size == TEXT_VOCAB_SIZE, (
            f"the model sizes its text vocabulary at {size}, not {TEXT_VOCAB_SIZE}: "
            f"the ids {TAG_IDS[0]}-{TAG_IDS[-1]} the tags occupy would not fit"
        )

    def test_the_tokenizer_is_built_from_the_pack_directory(self) -> None:
        """The tokenizer comes from the pack, so it carries the pack's ids."""
        source = _tokenizer_source_argument(_turbo_tree())
        assert source == "ckpt_dir", (
            "the tokenizer is no longer built from the pack directory "
            f"(found {source!r}); the pack's added tokens are what carry the "
            "tag ids"
        )

    def test_the_tokenizer_length_is_checked_against_the_vocabulary_size(self) -> None:
        """A tokenizer of the wrong length is a loud warning, not a surprise."""
        guard = _tokenizer_length_guard(_turbo_tree())
        assert guard == TEXT_VOCAB_SIZE, (
            f"the tokenizer length is compared against {guard}, not "
            f"{TEXT_VOCAB_SIZE}: a tokenizer that lost its added tokens would "
            "go unnoticed"
        )

    def test_the_text_reaches_the_tokenizer_as_it_was_typed(self) -> None:
        """The tag is in the text, so the text is what the tokenizer sees."""
        argument = _tokenizer_call_argument(_turbo_tree())
        assert argument == "text", (
            f"the tokenizer is handed {argument!r}, not the text: the bracketed "
            "tag would never reach the tokenizer that carries its id"
        )

    def test_the_vendored_tree_injects_no_tokens_of_its_own(self) -> None:
        """The vocabulary is the pack's -- nothing is added to it locally.

        ``tokenizer.add_tokens(...)`` is the shape a local patch would take to
        invent the vocabulary. Upstream ships the 19 ids inside the pack's own
        tokenizer, so the vendored tree must not add a single token: that is
        what makes the feature work with no vendored patch at all.
        """
        additions = [
            node.func.attr
            for node in ast.walk(_turbo_tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"add_tokens", "add_special_tokens"}
        ]
        assert not additions, (
            f"the vendored tree adds tokens to the tokenizer ({additions}): the "
            "tag ids belong to the pack's tokenizer, not to a local patch"
        )

    @pytest.mark.parametrize("tag", sorted(UPSTREAM_TAG_IDS))
    def test_the_text_cleanup_keeps_the_tag_exactly_as_typed(self, tag: str) -> None:
        """``punc_norm`` runs before the tokenizer and must not touch a tag.

        The cleanup collapses whitespace, swaps a handful of punctuation marks
        and appends a full stop. It must leave the bracketed tag alone -- the
        brackets and the spelling -- because the tokenizer matches added tokens
        case-sensitively and exactly.
        """
        cleaned = _punc_norm()(f"Hi {tag} there")
        assert tag in cleaned, (
            f"the vendored text cleanup changed {tag!r} into {cleaned!r}: the "
            "tag would no longer land on its added-token id"
        )


# ---------------------------------------------------------------------------
# The id table this file owns
# ---------------------------------------------------------------------------


class TestTheUpstreamTable:
    """The table is upstream's, transcribed here so a change is visible."""

    def test_the_table_holds_exactly_19_tags(self) -> None:
        assert len(UPSTREAM_TAG_IDS) == 19

    def test_the_ids_are_the_contiguous_block_below_the_vocabulary_size(self) -> None:
        assert sorted(UPSTREAM_TAG_IDS.values()) == list(TAG_IDS), (
            "the 19 tags do not occupy 50257-50275 as upstream published them"
        )
        assert max(UPSTREAM_TAG_IDS.values()) < TEXT_VOCAB_SIZE

    def test_the_planner_uses_the_same_19_spellings(self) -> None:
        """The node's vocabulary and the tokenizer's must be the same words.

        The planner keeps a tag in the text verbatim; the tokenizer only knows
        it under the exact spelling upstream published. A divergence here is a
        tag the node emits that the model cannot hear.
        """
        from ComfyUI_Chatterbox.modules.text_processing import TURBO_TAGS

        upstream = {tag[1:-1] for tag in UPSTREAM_TAG_IDS}
        assert set(TURBO_TAGS) == upstream, (
            "the planner's tag vocabulary and the tokenizer's added tokens "
            f"differ: only in {sorted(set(TURBO_TAGS) ^ upstream)}"
        )


# ---------------------------------------------------------------------------
# The env-gated half: the pack's own tokenizer
# ---------------------------------------------------------------------------


def _e2e_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "").strip() == "1"


@pytest.fixture(scope="module")
def pack_dir() -> str:
    """The pack directory the run reads, named through the environment.

    **No machine path is ever committed** -- the directory arrives through the
    environment, never as a constant in this file.
    """
    directory = os.environ.get(ENV_MODEL_DIR, "").strip()
    if not directory:
        pytest.skip(f"{ENV_MODEL_DIR} is not set: no pack to read a tokenizer from")
    if not os.path.isdir(directory):
        pytest.skip(f"{ENV_MODEL_DIR} points at {directory!r}, which does not exist")
    return directory


@pytest.fixture(scope="module")
def tokenizer(pack_dir: str):
    """The pack's own tokenizer, loaded the way the model loads it.

    ``transformers`` is not a declared dependency of this package -- it comes
    with ComfyUI, and the vendored tree imports it -- so it is imported here,
    inside the opt-in, and never at collection time.
    """
    transformers = pytest.importorskip(
        "transformers", reason="transformers is not installed in this interpreter"
    )
    return transformers.AutoTokenizer.from_pretrained(pack_dir, local_files_only=True)


_SKIP_REASON = (
    f"{ENV_ENABLE}=1 is not set: reading a real tokenizer needs a downloaded "
    f"pack, so the env-gated half of the proof stays skipped"
)


@pytest.mark.skipif(not _e2e_enabled(), reason=_SKIP_REASON)
class TestTheRealTokenizer:
    """The pack's own tokenizer, asked about every tag.

    This is the half that cannot be faked: it loads the tokenizer the model
    will load and asks it for the ids.
    """

    def test_the_tokenizer_is_sized_for_the_tag_ids(self, tokenizer) -> None:
        assert len(tokenizer) == TEXT_VOCAB_SIZE, (
            f"the pack's tokenizer holds {len(tokenizer)} entries, not "
            f"{TEXT_VOCAB_SIZE}: its added tokens are missing"
        )

    def test_the_pack_publishes_the_same_added_tokens(self, pack_dir: str) -> None:
        """The pack's own ``added_tokens.json`` agrees with the table above."""
        path = os.path.join(pack_dir, "added_tokens.json")
        if not os.path.isfile(path):
            pytest.skip(f"{path} is not in this pack directory")
        with open(path, "r", encoding="utf-8") as handle:
            published = json.load(handle)
        missing = [tag for tag in UPSTREAM_TAG_IDS if tag not in published]
        assert not missing, f"the pack's added_tokens.json does not carry {missing}"
        moved = {
            tag: published[tag]
            for tag in UPSTREAM_TAG_IDS
            if published[tag] != UPSTREAM_TAG_IDS[tag]
        }
        assert not moved, f"the pack's added tokens moved: {moved}"

    @pytest.mark.parametrize("tag", sorted(UPSTREAM_TAG_IDS))
    def test_the_tag_lands_on_its_upstream_id(self, tokenizer, tag: str) -> None:
        expected = UPSTREAM_TAG_IDS[tag]
        ids = tokenizer(tag).input_ids
        assert ids == [expected], (
            f"{tag!r} tokenized to {ids}, not [{expected}]: the tag would not "
            "be heard as the paralinguistic cue upstream trained it for"
        )

    def test_a_tag_inside_a_sentence_keeps_its_id(self, tokenizer) -> None:
        """The tag is not swallowed by the words around it."""
        ids = tokenizer("Hi [happy] there").input_ids
        assert ids == [17250, 220, 50265, 612], (
            f"'Hi [happy] there' tokenized to {ids}: the tag did not stay a "
            "single id inside the sentence"
        )

    def test_the_uppercase_spelling_is_a_different_token(self, tokenizer) -> None:
        """The tokenizer is case-sensitive, so ``[HAPPY]`` is not ``[happy]``.

        The planner recognises a tag case-insensitively but keeps the text as
        it was typed -- which is right: re-spelling the tag would invent a
        token upstream never trained.
        """
        ids = tokenizer("[HAPPY]").input_ids
        assert UPSTREAM_TAG_IDS["[happy]"] not in ids, (
            "[HAPPY] landed on the id of [happy]; the tokenizer is "
            "case-sensitive and the two are different tokens"
        )
        assert len(ids) > 1, "[HAPPY] was tokenized as one id; it is not a tag"


@pytest.mark.skipif(not _e2e_enabled(), reason=_SKIP_REASON)
class TestTheVariantCoverage:
    """Which family this pack directory can speak for.

    Turbo and Nano ship the same tokenizer vocabulary but different T3
    weights. A directory proves a family only when it holds that family's
    weights; a directory with neither proves the tokenizer and nothing else,
    and says so by name.
    """

    @pytest.mark.parametrize(
        "variant, weight_file", VARIANTS, ids=[variant for variant, _ in VARIANTS]
    )
    def test_the_family_this_pack_can_speak_for(
        self, pack_dir: str, tokenizer, variant: str, weight_file: str
    ) -> None:
        if not os.path.isfile(os.path.join(pack_dir, weight_file)):
            pytest.skip(
                f"{pack_dir} does not hold {weight_file}: this run proves "
                f"nothing about the {variant} variant"
            )
        assert len(tokenizer) == TEXT_VOCAB_SIZE
        for tag, expected in UPSTREAM_TAG_IDS.items():
            ids = tokenizer(tag).input_ids
            assert ids == [expected], (
                f"the {variant} pack tokenized {tag!r} to {ids}, not [{expected}]"
            )
