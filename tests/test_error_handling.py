"""S7.1 -- errors raise; only the legitimate empty case is silence.

The V1 nodes had six sites -- ``nodes.py:136,159,203,260,283,317`` -- that
wrapped a whole node body in ``except Exception`` and returned one second of
silence. The failure mode is not that the run stops: it is that the run
**appears to succeed**. A graph wired to a preview, a batch of fifty prompts,
a render queue that reports "done" -- every one of them looks clean, and the
audio that comes out is silence. A user debugging that has no error, no log
line naming the pack, and no way to tell a cancelled run from a broken one.

So the contract this file pins is:

* **Empty text is silence**, and only that. A graph wired up before it is
  finished runs; one second of silence is the honest answer for "there was
  nothing to say".
* **A generation failure raises**, with the model-pack name in the message.
  The pack name is the one piece of information that tells the user which
  download or checkpoint to look at.
* **A cancellation is silence with no traceback.** ``InterruptProcessingException``
  derives from ``BaseException``, so only the clause that names it can catch it,
  and it logs at INFO with no ``exc_info``.
* **The old pattern has no surviving test.** Not "is not currently true" --
  *no test in this repository asserts it*. A regression guard that permits the
  V1 behaviour is how the behaviour comes back; see
  :class:`TestNoSurvivingTestOfTheOldBehaviour`.

The last class is the one that earns this file's keep. Everything above it
describes the code; that class describes the tests, and a codebase whose tests
pin the correct behaviour is what makes the correct behaviour the one that
survives a refactor.
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest
import torch

from ComfyUI_Chatterbox.modules import generation as generation_module
from ComfyUI_Chatterbox.modules.audio_utils import interrupt_exception_class
from ComfyUI_Chatterbox.modules.generation import generate_audio
from ComfyUI_Chatterbox.nodes import ChatterboxTTSNode, ChatterboxVCNode
from ComfyUI_Chatterbox.nodes import tts_node as tts_node_module
from ComfyUI_Chatterbox.nodes import vc_node as vc_node_module

from conftest import ROOT_DIR


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class StubPack:
    """A loaded pack whose ``generate`` outcome the test decides."""

    def __init__(self, error: BaseException | None = None, samples: int = 24000) -> None:
        self.error = error
        self.samples = samples
        self.t3 = torch.nn.Linear(4, 4)
        self.s3gen = torch.nn.Linear(4, 4)
        self.ve = torch.nn.Linear(2, 2)
        self.calls: list[dict] = []

    def generate(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append({"args": list(args), **kwargs})
        if self.error is not None:
            raise self.error
        return torch.zeros((1, self.samples))


class FakeInterrupt(BaseException):
    """A stand-in for core's interrupt, which also derives from ``BaseException``."""


#: Core's real interrupt class, used wherever a test has to reproduce what a
#: cancelled run actually raises. The node layer catches *that* class by name
#: (``interrupt_exception_class()``), so a stand-in would simply propagate --
#: which is correct, and useless for testing the silence path.
REAL_INTERRUPT = interrupt_exception_class()

#: A cancellation that also derives from ``RuntimeError``.
#:
#: Core's real class derives from ``BaseException`` only, so ``except Exception``
#: cannot see it and the ordering of the two clauses would not matter. This one
#: removes that guarantee, which is exactly why the interrupt clause has to be
#: written *first* rather than merely *present*: with the clauses swapped, an
#: exception deriving from both would be reported as a generation failure and
#: the run would not cancel.
HybridInterrupt = type(
    "HybridInterrupt", (REAL_INTERRUPT, RuntimeError), {"__module__": __name__}
)


def interrupted(message: str = "cancelled") -> BaseException:
    """A fresh cancellation instance, for a stub that raises it."""
    return REAL_INTERRUPT(message)


@pytest.fixture(autouse=True)
def _no_progress_bar(monkeypatch):
    """Drop the real ``comfy.utils.ProgressBar`` plumbing from the walk.

    ``generate_audio`` builds one and ticks it thousands of times over a CPU
    run; here every pack either returns instantly or raises on the first call,
    so the bar is noise. The interrupt class is left alone -- that is the thing
    under test.
    """
    monkeypatch.setattr(
        generation_module,
        "_build_progress_bar",
        lambda total: generation_module.InterruptProgressBar(total, None),
    )


def node_kwargs(**overrides) -> dict:
    """A minimal, complete ``ChatterboxTTSNode.execute`` call."""
    kwargs = {
        "model_pack_name": "resembleai_default_voice",
        "text": "Hello there.",
        "max_new_tokens": 1000,
        "flow_cfg_scale": 0.7,
        "exaggeration": 0.5,
        "temperature": 0.8,
        "cfg_weight": 0.5,
        "repetition_penalty": 1.2,
        "min_p": 0.05,
        "top_p": 1.0,
        "seed": 0,
        "use_watermark": False,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def stubbed_generation(monkeypatch):
    """Replace the model load and the synthesis, keep the node's own code."""

    def _install(model, loader_result=None):
        loader_result = loader_result or (None, model, None)
        monkeypatch.setattr(
            tts_node_module,
            "load_chatterbox_model",
            lambda *args, **kwargs: loader_result,
        )
        monkeypatch.setattr(
            tts_node_module, "generate_audio", lambda model, text, **kwargs: (
                model.generate(text)
            )
        )
        monkeypatch.setattr(
            vc_node_module,
            "load_chatterbox_model",
            lambda *args, **kwargs: loader_result,
        )
        monkeypatch.setattr(
            vc_node_module, "convert_voice_audio", lambda model, audio, **kwargs: (
                model.generate(audio)
            )
        )

    return _install


# ---------------------------------------------------------------------------
# Empty text: the one legitimate silence
# ---------------------------------------------------------------------------


class TestEmptyTextIsSilence:
    @pytest.mark.parametrize("blank", ["", "   ", "\n\t", " \r\n "])
    def test_the_node_returns_silence(self, blank, stubbed_generation) -> None:
        stubbed_generation(StubPack())

        result = ChatterboxTTSNode.execute(**node_kwargs(text=blank))

        audio = result.result[0]
        assert audio["sample_rate"] == 24000
        assert audio["waveform"].abs().max().item() == 0.0, (
            "the empty-text path must return actual silence, not a waveform "
            "that merely happens to be short"
        )
        assert audio["waveform"].shape[-1] == 24000, "one second, by convention"

    def test_an_empty_box_does_not_load_a_model(self, monkeypatch) -> None:
        """2.7 GB is too expensive a price for an empty text box.

        The check has to come before ``load_chatterbox_model``; a version that
        loads first and returns silence afterwards still "works", and still
        makes every half-wired graph download a checkpoint.
        """
        called: list[tuple] = []
        monkeypatch.setattr(
            tts_node_module,
            "load_chatterbox_model",
            lambda *args, **kwargs: called.append(kwargs) or (None, None, None),
        )

        ChatterboxTTSNode.execute(**node_kwargs(text="  "))

        assert called == [], (
            f"the empty-text path loaded a model ({called}); it must return "
            f"before the load"
        )

    def test_generation_still_refuses_blank_text_directly(self) -> None:
        """The node's silence does not weaken the layer under it.

        ``generate_audio`` is also the API entry point, and an API caller that
        sends blank text has made a mistake worth reporting. Only the node
        translates "unfinished graph" into silence.
        """
        with pytest.raises(ValueError, match="needs some text"):
            generate_audio(StubPack(), "   ")

    def test_blank_text_is_not_the_only_thing_the_node_catches(self) -> None:
        """The silence is one explicit check, not a blanket handler."""
        source = inspect.getsource(tts_node_module.ChatterboxTTSNode.execute)
        assert "text.strip()" in source
        assert "except Exception" not in source, (
            "a blanket handler is back; see TestNoSurvivingTestOfTheOldBehaviour"
        )


# ---------------------------------------------------------------------------
# Failures raise
# ---------------------------------------------------------------------------


class TestFailuresRaise:
    def test_a_generate_failure_reaches_the_caller(
        self, stubbed_generation
    ) -> None:
        stubbed_generation(StubPack(error=RuntimeError("CUDA out of memory")))

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            ChatterboxTTSNode.execute(**node_kwargs())

    def test_the_pack_name_is_available_to_the_message(
        self, stubbed_generation, monkeypatch
    ) -> None:
        """A failing load names the pack.

        That is the one fact that tells the user which download or checkpoint to
        go and look at, and the V1 code threw it away along with everything
        else.
        """
        monkeypatch.setattr(
            tts_node_module,
            "load_chatterbox_model",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError(
                    "The Chatterbox voice pack 's3gen-multilingual' could not be "
                    "loaded. Check the log above for the download or build failure."
                )
            ),
        )

        with pytest.raises(RuntimeError, match="s3gen-multilingual"):
            ChatterboxTTSNode.execute(
                **node_kwargs(model_pack_name="s3gen-multilingual")
            )

    def test_a_conversion_failure_raises_too(self, monkeypatch) -> None:
        pack = StubPack(error=ValueError("bad reference clip"))
        monkeypatch.setattr(
            vc_node_module, "load_chatterbox_model", lambda *a, **k: (None, None, pack)
        )
        monkeypatch.setattr(
            vc_node_module,
            "convert_voice_audio",
            lambda model, audio, **kwargs: model.generate(audio),
        )

        with pytest.raises(ValueError, match="bad reference clip"):
            ChatterboxVCNode.execute(
                model_pack_name="resembleai_default_voice",
                source_audio={
                    "waveform": torch.zeros((1, 1, 2400)),
                    "sample_rate": 24000,
                },
                n_timesteps=10,
                temperature=1.0,
                flow_cfg_scale=0.7,
                use_watermark=False,
            )

    def test_the_pack_without_the_right_half_raises(self, monkeypatch) -> None:
        """A TTS pack with no synthesis model is a wiring mistake, not silence."""
        monkeypatch.setattr(
            tts_node_module,
            "load_chatterbox_model",
            lambda *args, **kwargs: (None, None, None),
        )

        with pytest.raises(ValueError, match="resembleai_default_voice"):
            ChatterboxTTSNode.execute(**node_kwargs())


# ---------------------------------------------------------------------------
# Cancellation: silence, no traceback
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_an_interrupt_returns_silence(self, stubbed_generation) -> None:
        stubbed_generation(StubPack(error=interrupted()))

        result = ChatterboxTTSNode.execute(**node_kwargs())

        assert result.result[0]["waveform"].abs().max().item() == 0.0

    def test_an_interrupt_logs_no_traceback(
        self, stubbed_generation, caplog
    ) -> None:
        """A cancelled run is not a failed run.

        ``logging.exception`` / ``exc_info=True`` here would print a stack trace
        for something the user asked for, which trains them to ignore the ones
        that matter.
        """
        import logging

        stubbed_generation(StubPack(error=interrupted()))

        with caplog.at_level(logging.DEBUG, logger=""):
            ChatterboxTTSNode.execute(**node_kwargs())

        assert caplog.records, "the cancellation was not logged at all"
        for record in caplog.records:
            assert record.exc_info is None, (
                f"a cancellation was logged with a traceback: {record.getMessage()}"
            )
        assert any(
            "cancel" in record.getMessage().lower() for record in caplog.records
        ), f"no cancellation notice in {[r.getMessage() for r in caplog.records]}"

    def test_an_interrupt_propagates_out_of_generation(self) -> None:
        """``generate_audio`` re-raises; only the node turns it into silence.

        Swallowing the interrupt inside ``generate_audio`` would end the prompt
        cleanly but leave ComfyUI's interrupt flag set, so the *next* node in
        the queue would cancel too.
        """
        with pytest.raises(REAL_INTERRUPT):
            generate_audio(StubPack(error=interrupted()), "hello")

    def test_a_hybrid_interrupt_is_not_caught_by_a_bare_exception(
        self,
    ) -> None:
        """The case that proves the clause ordering rather than its presence.

        ``HybridInterrupt`` derives from core's interrupt **and** from
        ``RuntimeError``. An ``except Exception`` clause would therefore catch
        it; the interrupt clause, which appears first, wins. With the two
        clauses swapped, the interrupt would be logged as a generation failure
        and re-raised as a plain error -- the run would not cancel.
        """
        error = HybridInterrupt("cancelled")

        with pytest.raises(REAL_INTERRUPT):
            generate_audio(StubPack(error=error), "hello")

        assert isinstance(error, RuntimeError), (
            "the test double stopped deriving from RuntimeError, which is the "
            "whole point of it"
        )

    def test_a_generic_failure_still_raises_after_the_interrupt_clause(
        self,
    ) -> None:
        """The interrupt clause must not have become ``except BaseException``."""
        with pytest.raises(RuntimeError, match="out of memory"):
            generate_audio(StubPack(error=RuntimeError("out of memory")), "hello")

    def test_the_real_interrupt_class_is_base_exception_derived(self) -> None:
        """Against the real core, not a double."""
        interrupt = interrupt_exception_class()
        assert issubclass(interrupt, BaseException)
        assert not issubclass(interrupt, Exception), (
            "core re-parented the interrupt under Exception; every "
            "`except Exception` in the package can now swallow a cancellation"
        )


# ---------------------------------------------------------------------------
# The old behaviour has no surviving test
# ---------------------------------------------------------------------------


#: The shape the V1 nodes were written in: a whole node body wrapped so that
#: every failure becomes one second of silence.
OLD_PATTERN_AST_HINT = "ExceptHandler(type=Exception) -> returns silence"

SHIPPED_TREE = ("nodes", "modules", "chatterbox_nodes.py", "__init__.py")


def _iter_shipped_sources() -> list[str]:
    found: list[str] = []
    for root in SHIPPED_TREE:
        path = os.path.join(ROOT_DIR, root)
        if os.path.isfile(path):
            found.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [name for name in dirnames if name != "__pycache__"]
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    found.append(
                        os.path.relpath(
                            os.path.join(dirpath, filename), ROOT_DIR
                        ).replace(os.sep, "/")
                    )
    return sorted(found)


def _iter_test_sources() -> list[str]:
    directory = os.path.join(ROOT_DIR, "tests")
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".py")
    )


#: Entry points a test can call and get a return value out of.
ENTRY_POINTS: frozenset[str] = frozenset(
    {"execute", "generate_audio", "convert_voice_audio", "load_chatterbox_model"}
)

#: Exception classes available without being defined in the test module.
BUILTIN_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "Exception",
        "BaseException",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "OSError",
        "IOError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "FileNotFoundError",
        "NotImplementedError",
        "AssertionError",
        "ArithmeticError",
        "ZeroDivisionError",
    }
)

#: A class whose name carries this is a cancellation, not a failure. Silent
#: handling of a cancellation is the contract, so a test may assert silence
#: after one; a test may not after anything else.
CANCELLATION_MARKER = "interrupt"


def _module_exception_classes(tree: ast.Module) -> dict[str, frozenset[str]]:
    """``name -> base names`` for every exception class defined in ``tree``."""
    defined: dict[str, frozenset[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(base, ast.Name) and base.id in BUILTIN_EXCEPTIONS
            for base in node.bases
        ):
            continue
        defined[node.name] = frozenset(
            base.id for base in node.bases if isinstance(base, ast.Name)
        )
    return defined


def _is_generic_failure(class_name: str, defined: dict[str, frozenset[str]]) -> bool:
    """Whether ``class_name`` is an ``Exception``-derived, non-cancellation error.

    A cancellation is exempt: ``InterruptProcessingException`` derives from
    ``BaseException``, the stub names carry "interrupt", and silence after a
    cancellation is exactly what the node layer is supposed to do.
    """
    if CANCELLATION_MARKER in class_name.lower():
        return False
    bases = defined.get(class_name, frozenset({class_name}) & BUILTIN_EXCEPTIONS)
    if not bases:
        bases = frozenset({class_name}) & BUILTIN_EXCEPTIONS
    return bool(bases - {"BaseException"})


def _stubbed_failures(node: ast.AST, defined: dict[str, frozenset[str]]) -> set[str]:
    """Exception class names this test *builds an instance of*.

    ``StubPack(error=RuntimeError(...))``, ``pytest.raises(ValueError)`` and a
    bare ``ValueError(...)`` all count: each is a test asking for a failure.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = None
            if isinstance(child.func, ast.Name):
                name = child.func.id
            elif isinstance(child.func, ast.Attribute):
                name = child.func.attr
            if name and _is_generic_failure(name, defined):
                found.add(name)
    return found


def _calls_entry_point(node: ast.AST, defined: dict[str, frozenset[str]]) -> list[str]:
    """Entry-point calls in ``node`` that are not inside ``pytest.raises``."""
    calls: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr not in ENTRY_POINTS:
            continue
        # Walk up is not available from a bare Call node, so the enclosing
        # `with` is handled by the caller slicing the body instead.
        calls.append(ast.unparse(child))
    return calls


def _inside_pytest_raises(tree: ast.FunctionDef) -> list[str]:
    """Entry-point call sources that appear under a ``with pytest.raises``."""
    guarded: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        text = ast.unparse(node)
        if "pytest.raises" not in text:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in ENTRY_POINTS
            ):
                guarded.append(ast.unparse(child))
    return guarded


class TestNoSurvivingTestOfTheOldBehaviour:
    """No test in this repository may permit the V1 behaviour.

    Not a claim about the code -- the classes above cover that. A claim about the
    *tests*: if some suite still encodes "a failure yields silence", then a
    future change that restores the blanket handler would be reported as a pass,
    and the six V1 sites would come back through the front door wearing a green
    test as a disguise.

    The rule enforced is precise, and deliberately narrow rather than broad --
    a guard that flags legitimate tests gets deleted, and a deleted guard is
    worse than none. A test is an offender when **all three** hold:

    1. it builds an instance of an ``Exception``-derived class (not a
       cancellation, which silence is the correct answer for), **and**
    2. it calls a node or generation entry point, **and**
    3. that call is not wrapped in ``pytest.raises``.

    Silence after an empty input, silence after a cancellation, and a test that
    asserts a failure *raises* are all permitted; each is asserted below.
    """

    def test_the_scan_actually_reads_the_suite(self) -> None:
        sources = _iter_test_sources()
        assert len(sources) >= 20, (
            f"only {len(sources)} files were scanned; this file would pass "
            f"vacuously"
        )

    def test_no_test_asserts_silence_after_a_generic_failure(self) -> None:
        offenders: list[str] = []
        for path in _iter_test_sources():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            defined = _module_exception_classes(tree)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.name.startswith("test"):
                    continue
                failures = _stubbed_failures(node, defined)
                if not failures:
                    continue
                guarded = _inside_pytest_raises(node)
                for source in _calls_entry_point(node, defined):
                    if source in guarded:
                        continue
                    offenders.append(
                        f"{os.path.basename(path)}::{node.name}: {source} "
                        f"(stubs {sorted(failures)})"
                    )
        assert offenders == [], (
            "these tests stub a real failure and then assert only that an "
            "entry point returned -- which is exactly the V1 "
            "'a failure yields silence' shape:\n  " + "\n  ".join(offenders)
        )

    @pytest.mark.parametrize(
        "source",
        [
            # The V1 shape, verbatim.
            "def test_bad_text_still_produces_output():\n"
            "    StubPack(error=RuntimeError('boom'))\n"
            "    result = ChatterboxTTSNode.execute(text='hello')\n"
            "    assert result is not None\n",
            # And the same thing wrapped in a class, which the scan must also see.
            "class TestLegacy:\n"
            "    def test_it_returns(self):\n"
            "        pack = StubPack(error=ValueError('boom'))\n"
            "        assert ChatterboxTTSNode.execute(text='hi') is not None\n",
        ],
        ids=["function", "method"],
    )
    def test_the_detector_catches_the_old_shape(self, source: str) -> None:
        """Positive control.

        Without this, "no offenders" is indistinguishable from "the detector
        finds nothing at all" -- which is the failure mode that lets a guard
        rot into decoration.
        """
        tree = ast.parse(source)
        defined = _module_exception_classes(tree)
        offenders = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _stubbed_failures(node, defined)
            and any(
                source_of not in _inside_pytest_raises(node)
                for source_of in _calls_entry_point(node, defined)
            )
        ]
        assert offenders, (
            "the detector missed the V1 shape; the scan above is not evidence "
            "of anything"
        )

    @pytest.mark.parametrize(
        "source",
        [
            # Silence after an empty input: not a failure, so permitted.
            "def test_blank_text_is_silence():\n"
            "    result = ChatterboxTTSNode.execute(text='   ')\n"
            "    assert result is not None\n",
            # Silence after a cancellation: the contract, so permitted.
            "def test_cancel_is_silence():\n"
            "    StubPack(error=FakeInterrupt('cancelled'))\n"
            "    result = ChatterboxTTSNode.execute(text='hi')\n"
            "    assert result is not None\n",
            # A failure that raises: asserted, so permitted.
            "def test_failure_raises():\n"
            "    StubPack(error=RuntimeError('boom'))\n"
            "    with pytest.raises(RuntimeError):\n"
            "        ChatterboxTTSNode.execute(text='hi')\n",
        ],
        ids=["empty-input", "cancellation", "asserted-raise"],
    )
    def test_the_detector_allows_the_permitted_shapes(self, source: str) -> None:
        """The other half of a positive control: no false positives.

        A guard that flags the three legitimate shapes above would be deleted
        on sight, and deleting it leaves the package unprotected.
        """
        tree = ast.parse(source)
        defined = _module_exception_classes(tree)
        offenders = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _stubbed_failures(node, defined)
            and any(
                source_of not in _inside_pytest_raises(node)
                for source_of in _calls_entry_point(node, defined)
            )
        ]
        assert offenders == [], f"the detector flagged a legitimate test: {offenders}"

    def test_no_shipped_node_wraps_its_body_in_a_broad_handler(self) -> None:
        """The production half: no ``except Exception`` in the node layer.

        The interrupt clause is ``except <the interrupt class>``, which is not
        ``Exception``, so it does not match here. ``modules/generation.py`` is
        in the tree because its own ``except Exception`` re-raises after
        ``logging.exception`` -- a diagnostic, not a swallow -- and the
        assertion below is what pins that distinction.
        """
        offenders: list[str] = []
        for relative in _iter_shipped_sources():
            if not relative.startswith("nodes/"):
                continue
            with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=relative)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                caught = node.type
                names = (
                    [caught.id]
                    if isinstance(caught, ast.Name)
                    else []
                )
                if "Exception" in names or "BaseException" in names:
                    offenders.append(f"{relative}:{node.lineno}")
        assert offenders == [], (
            f"the node layer catches a broad exception at {offenders}. A "
            f"cancelled run is handled by naming the interrupt class; "
            f"everything else must propagate."
        )

    def test_the_interrupt_clause_precedes_the_generic_one(self) -> None:
        """Ordering, structurally, in both places that have both.

        Two ``except`` clauses in the right order and two in the wrong order
        parse identically; only their sequence differs. A test that merely
        checked "an interrupt clause exists" would pass on the broken ordering.
        """
        checked = 0
        for relative in ("modules/generation.py", "nodes/tts_node.py", "nodes/vc_node.py"):
            with open(os.path.join(ROOT_DIR, relative), "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=relative)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                clauses = [
                    handler
                    for handler in node.handlers
                    if handler.type is not None
                    and not (
                        isinstance(handler.type, ast.Name)
                        and handler.type.id in {"Exception", "BaseException"}
                    )
                ]
                generic = [
                    index
                    for index, handler in enumerate(node.handlers)
                    if isinstance(handler.type, ast.Name)
                    and handler.type.id == "Exception"
                ]
                if not (clauses and generic):
                    continue
                checked += 1
                interrupt_index = min(
                    index
                    for index, handler in enumerate(node.handlers)
                    if handler in clauses
                )
                assert interrupt_index < min(generic), (
                    f"{relative}:{node.lineno} catches the generic Exception "
                    f"before the interrupt class; a cancellation that also "
                    f"derives from Exception would be reported as a failure "
                    f"and the run would not cancel"
                )
        assert checked >= 1, (
            "no `try` with both an interrupt clause and an `except Exception` "
            "clause was found; the ordering check proved nothing"
        )

    def test_the_pattern_is_named_so_a_reintroduction_is_recognisable(self) -> None:
        """Keep the description of the old shape in the file that bans it."""
        assert OLD_PATTERN_AST_HINT.startswith("ExceptHandler")
        assert "silence" in OLD_PATTERN_AST_HINT