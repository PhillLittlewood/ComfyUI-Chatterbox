"""S9.5 -- nothing that ships may reference the local scratch corpus.

This repository was written against a private corpus that is symlinked into the
checkout under a dot-directory (named by the ``DEV_DIR`` constant below, spelled
out nowhere in this file -- see *the wrinkle*, further down). That directory
exists on exactly one machine. A path pointing into it is not a path, it is a
claim about the author's desktop, and it is broken everywhere else: the reader
gets a dangling reference instead of a file.

So three things are banned from anything git would ship:

* the dot-directory itself, by name;
* the reference corpus' plan directory, by name;
* any absolute path pointing outside the repository -- a drive letter, a home
  directory, an MSYS-style drive root.

**Why a test and not a review note.** Because the failure is invisible to every
tool the build already runs. Nothing fails when a comment grows a path; the
damage only shows up on someone else's checkout, in a doc link, months later.

**The file set is git's, not ours.** Tracked files *and* untracked-but-not-ignored
files are both scanned, so the check works while a tree is still being built --
which is when a local path is most likely to get pasted in. Files git already
ignores are out of scope by definition: they are not shipping.

**The wrinkle: this file, and its neighbours.** A scanner must spell out what it
scans for, so this file, ``tests/test_vendor_patch_manifest.py`` and
``tests/test_no_dead_modules.py`` all contain the forbidden strings as *data*.
Excluding them wholesale would be a hole big enough to walk a real leak through,
so instead each exclusion is an :class:`Allowance`: a named file, a written
reason, and a **hard ceiling on how many lines it may be exempt for**. The
exemption stops existing the moment the file grows past its ceiling -- which is
the moment it stops being a scanner quoting a pattern and starts being prose
that happens to mention one. ``TestTheAllowlistStaysNarrow`` is what enforces
that.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

import pytest

from conftest import ROOT_DIR

# The banned name, assembled from pieces so that this module's own source does
# not contain it as a contiguous literal. That keeps the scanner honest against
# itself without an allowance for its own pattern table.
DEV_DIR = "." + "dev"

# `plans/` -- the reference corpus' plan directory. Same treatment.
PLANS_DIR = "plans" + "/"

# `<something>.<ext>` -- an extension anchor, so a `.dev` inside a longer word
# is not a hit. Preceded by nothing or by a non-alphanumeric, which is what
# keeps the upstream `pdm.fming.dev/` URL in .gitignore out of the results.
DOT_DIR = re.compile(r"(?<![A-Za-z0-9])" + re.escape(DEV_DIR) + r"(?![A-Za-z0-9])")
PLANS_PATH = re.compile(re.escape(PLANS_DIR))

# `C:\` or `C:/`. The lookbehind excludes URL schemes: in `https:/` the `s` is
# alphanumeric, so it is not mistaken for a drive letter.
WINDOWS_DRIVE = re.compile(r"(?<![A-Za-z0-9_.\-])[A-Za-z]:[\\/]")

# A home or system root: the shapes that only ever mean "somewhere else".
POSIX_ROOT = re.compile(r"/(?:home|Users|root|opt|mnt|media)/")

# An MSYS/Git-Bash drive root, e.g. `/c/_Dev/...`. The trailing underscore
# requirement is what keeps ordinary relative-looking paths out of the results.
MSYS_DRIVE = re.compile(r"(?<![A-Za-z0-9_])/[A-Za-z]/_")


@dataclass(frozen=True)
class Rule:
    """One banned shape, with the explanation a failure message should carry."""

    name: str
    pattern: re.Pattern[str]
    why: str


RULES = (
    Rule("scratch-directory", DOT_DIR, f"{DEV_DIR}/ exists only on the author's machine"),
    Rule("corpus-plans-directory", PLANS_PATH, f"{PLANS_DIR} points into the private reference corpus"),
    Rule("windows-absolute-path", WINDOWS_DRIVE, "a drive letter is a path off this machine"),
    Rule("posix-absolute-path", POSIX_ROOT, "a home directory is a path off this machine"),
    Rule("msys-drive-root", MSYS_DRIVE, "an MSYS drive root is a path off this machine"),
)


@dataclass(frozen=True)
class Allowance:
    """A file permitted to mention what the scan forbids, within a budget.

    ``max_lines`` is the point of the class. An exclusion without a ceiling is
    a hole; with one, the exclusion expires on its own the first time the file
    starts using the name for real instead of naming the rule.
    """

    path: str
    reason: str
    max_lines: int


ALLOWANCES: tuple[Allowance, ...] = (
    Allowance(
        path="tests/test_no_dev_references.py",
        reason=(
            "This module is the scanner: the banned shapes have to be written "
            "out literally here to be searched for at all."
        ),
        max_lines=20,
    ),
    Allowance(
        path="tests/test_vendor_patch_manifest.py",
        reason=(
            "Its own leak scanner quotes the same needles -- a test is allowed "
            "to mention the thing it forbids."
        ),
        max_lines=4,
    ),
    Allowance(
        path="tests/test_vendor_integrity.py",
        reason=(
            "Same shape: it asserts the vendored tree carries no home-directory "
            "path, so it has to write the shapes it forbids."
        ),
        max_lines=2,
    ),
    Allowance(
        path="tests/test_no_dead_modules.py",
        reason=(
            "Its docstring names the dot-directories it skips, as examples; "
            "prose describing the rule is not a reference to the corpus."
        ),
        max_lines=4,
    ),
    Allowance(
        path=".gitignore",
        reason=(
            "The ignore pattern itself has to name the directory for the "
            "exclusion to take effect, and the comment explaining why explains "
            "it by name."
        ),
        max_lines=8,
    ),
)

ALLOWED = {allowance.path: allowance for allowance in ALLOWANCES}

# Anything larger than this is a binary or a data blob; neither is prose that
# can leak a path by accident, and reading them would cost seconds.
MAX_SCAN_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Violation:
    """One banned shape, found on one line of one file."""

    path: str
    line_number: int
    line: str
    rule: Rule

    def render(self) -> str:
        text = self.line.strip()
        if len(text) > 90:
            text = text[:87] + "..."
        return (
            f"{self.path}:{self.line_number}: [{self.rule.name}] {text}\n"
            f"    {self.rule.why}"
        )


# ---------------------------------------------------------------------------
# The file set -- git decides what ships, so git decides what we read
# ---------------------------------------------------------------------------


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def _is_git_checkout() -> bool:
    if shutil.which("git") is None:
        return False
    return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


def candidate_files() -> list[str]:
    """Every file that could ship, as repository-relative POSIX paths.

    Tracked files plus untracked-but-not-ignored files. Passing ``--exclude-standard``
    to the second query is what keeps build output and editor droppings out:
    those are already declared not to ship, so scanning them would only produce
    noise from files nobody will ever commit.
    """
    tracked = _git("ls-files", "-z").stdout
    untracked = _git("ls-files", "-z", "-o", "--exclude-standard").stdout
    paths = set()
    for stream in (tracked, untracked):
        for entry in stream.split("\0"):
            if entry:
                paths.add(entry.replace(os.sep, "/"))
    return sorted(paths)


def _read_text(path: str, base: str = ROOT_DIR) -> str | None:
    """The file's text, or ``None`` when it is binary, too big or unreadable."""
    absolute = os.path.join(base, path)
    try:
        if os.path.getsize(absolute) > MAX_SCAN_BYTES:
            return None
        with open(absolute, "rb") as handle:
            raw = handle.read()
    except OSError:  # pragma: no cover - a vanished or locked file
        return None
    if b"\0" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def violations_in(path: str, text: str) -> list[Violation]:
    """Every banned shape in one file's text, in line order."""
    found: list[Violation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            if rule.pattern.search(line):
                found.append(Violation(path, number, line, rule))
    return found


def exempt_lines(path: str, violations: list[Violation]) -> tuple[list[Violation], int]:
    """Split a file's violations into ``(shipped, exempt)``.

    The allowance applies per *line*, and only while the file stays inside its
    ceiling: past that, nothing is exempt and every hit is reported.
    """
    allowance = ALLOWED.get(path)
    if allowance is None:
        return violations, 0
    distinct = len({violation.line_number for violation in violations})
    if distinct > allowance.max_lines:
        return violations, 0
    return [], distinct


def scan(
    paths: list[str] | None = None, base: str = ROOT_DIR
) -> tuple[list[Violation], dict[str, int]]:
    """Run the whole scan.

    ``paths`` defaults to every file git would ship; ``base`` defaults to the
    repository root and exists so the planted-violation tests can point the same
    function at a temporary tree. Returns ``(shipped violations, exempt counts)``.
    """
    shipped: list[Violation] = []
    exempt: dict[str, int] = {}
    for path in paths if paths is not None else candidate_files():
        text = _read_text(path, base)
        if text is None:
            continue
        found = violations_in(path, text)
        if not found:
            continue
        remaining, count = exempt_lines(path, found)
        if count:
            exempt[path] = count
        shipped.extend(remaining)
    return shipped, exempt


# ---------------------------------------------------------------------------
# The scanner itself -- proven against planted violations
# ---------------------------------------------------------------------------


CLEAN_SOURCE = '"""A module that talks about models and nothing else."""\n'

PLANTED = {
    "scratch-directory": (
        f'SCRATCH = "{DEV_DIR}/plans"\n',
        f"The corpus lives at {DEV_DIR}/plans on the author's machine.",
    ),
    "corpus-plans-directory": (
        'PLAN = "reference/' + PLANS_DIR + 'step-one.md"\n',
        f"see {PLANS_DIR}step-one.md for the design",
    ),
    "windows-absolute-path": (
        'ROOT = "C:\\\\Users\\\\someone\\\\ComfyUI"\n',
        "the checkout lives at C:/Users/someone/ComfyUI",
    ),
    "posix-absolute-path": (
        'SRC = "/home/someone/reference/repos"\n',
        "borrowed from /Users/someone/reference/repos",
    ),
    "msys-drive-root": (
        'PY = "/c/_Dev/Python/.venv"\n',
        "run /c/_Dev/Python/.venv/Scripts/python.exe",
    ),
}

INNOCENT = {
    "a-url-with-dev-in-the-host": "https://pdm.fming.dev/#use-with-ide\n",
    "an-english-word": "# the device. the development stage. the devs.\n",
    "a-url-scheme": "https://example.invalid/model\n",
    "a-relative-path": "src/PATCHES.md and example_workflows/*.json\n",
    "a-repo-path": "ComfyUI/models/tts/chatterbox/resembleai_default_voice\n",
    "a-lone-slash": "ratio = 1 / 2 / 3\n",
}


class TestTheScannerDetectsPlantedViolations:
    """The scan is only worth its ratchet if it fires. Each rule gets a file
    that must trip it -- written to a temporary directory, never to the repo."""

    @pytest.mark.parametrize("rule_name", sorted(PLANTED))
    def test_a_planted_violation_is_reported(self, tmp_path, rule_name: str) -> None:
        rule = next(rule for rule in RULES if rule.name == rule_name)
        for text in PLANTED[rule_name]:
            found = [
                violation
                for violation in violations_in("planted.py", text)
                if violation.rule is rule
            ]
            assert found, (
                f"{rule.name} did not fire on {text!r}. If the pattern was "
                "weakened to make some existing file pass, this is where it "
                "should have been caught."
            )
            assert found[0].line_number >= 1
            assert rule.name in found[0].render()

    @pytest.mark.parametrize("name", sorted(INNOCENT))
    def test_an_innocent_line_is_not_reported(self, name: str) -> None:
        text = INNOCENT[name]
        assert not violations_in(name, text), (
            f"{name} ({text.strip()!r}) tripped a rule; the patterns are too "
            "broad and will drown real findings in false positives"
        )

    def test_a_clean_file_produces_nothing(self) -> None:
        assert violations_in("clean.py", CLEAN_SOURCE) == []

    def test_the_planted_scan_reports_every_rule_at_least_once(self, tmp_path) -> None:
        """End to end through :func:`scan`, not just the per-rule helper."""
        planted = []
        for index, texts in enumerate(PLANTED.values()):
            for inner, text in enumerate(texts):
                relative = f"tmp_planted/s{index}_{inner}.py"
                (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / relative).write_text(text, encoding="utf-8")
                planted.append(relative)
        found, exempt = scan(planted, base=str(tmp_path))
        reported = {violation.rule.name for violation in found}
        assert not exempt, f"planted files must not be allowlisted, got {exempt}"
        assert reported == {rule.name for rule in RULES}, (
            f"the scan reported {sorted(reported)}, expected every rule"
        )
        # At least one hit per planted file. Not exactly one per file: a line
        # can trip two rules at once, and that overlap is the scan working.
        unnoticed = [path for path in planted if not any(v.path == path for v in found)]
        assert not unnoticed, f"these planted files produced no hit at all: {unnoticed}"
        assert len(found) >= len(planted)

    def test_a_file_over_its_ceiling_stops_being_exempt(self, tmp_path) -> None:
        """The allowance is a budget, not a permanent pardon."""
        allowance = ALLOWED["tests/test_vendor_patch_manifest.py"]
        over = allowance.max_lines + 1
        lines = [f'# reference {DEV_DIR}/{i}' for i in range(over)]
        text = "\n".join(lines) + "\n"
        remaining, count = exempt_lines(allowance.path, violations_in(allowance.path, text))
        assert count == 0, "a file over its ceiling must lose its exemption"
        assert len(remaining) == over


class TestNothingShipsALocalReference:
    """The actual assertion. Everything git would ship, scanned."""

    @pytest.fixture(scope="class")
    def scanned(self) -> tuple[list[Violation], dict[str, int]]:
        return scan()

    def test_the_scan_read_a_meaningful_number_of_files(self) -> None:
        assert len(candidate_files()) > 50, (
            "the candidate set is implausibly small -- git is probably failing "
            "and this whole file would pass vacuously"
        )

    @pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.name)
    def test_no_shipped_file_contains_this(self, scanned, rule: Rule) -> None:
        shipped, exempt = scanned
        offenders = [violation for violation in shipped if violation.rule is rule]
        report = "\n".join(violation.render() for violation in offenders)
        context = (
            f"\n(exempt, inside its ceiling: {sorted(exempt)})" if exempt else ""
        )
        assert not offenders, (
            f"{len(offenders)} shipped line(s) match {rule.name}: "
            f"{rule.why}. Everything a user can install does not exist on this "
            f"machine.\n{report}{context}"
        )


class TestTheAllowlistStaysNarrow:
    """An allowlist is a hole with a comment on it. Keep it a small hole."""

    def test_every_allowance_carries_a_real_reason(self) -> None:
        for allowance in ALLOWANCES:
            assert len(allowance.reason) >= 40, (
                f"{allowance.path} is allowlisted with a reason too short to "
                "justify it; say why the mention is unavoidable"
            )

    def test_every_allowance_names_a_file_that_exists(self) -> None:
        for allowance in ALLOWANCES:
            assert os.path.exists(os.path.join(ROOT_DIR, allowance.path)), (
                f"{allowance.path} is allowlisted but no longer exists; drop the "
                "allowance so a future file at that path is scanned again"
            )

    def test_the_allowlist_is_small(self) -> None:
        assert len(ALLOWANCES) <= 6, (
            "an allowlist of "
            f"{len(ALLOWANCES)} entries is no longer a list of tests that have "
            "to mention what they forbid"
        )

    def test_every_allowance_is_inside_its_own_ceiling(self) -> None:
        """Otherwise the allowance is dead weight -- and, worse, the next file "
        "at that path inherits it silently."""
        for allowance in ALLOWANCES:
            text = _read_text(allowance.path)
            assert text is not None, f"{allowance.path} is binary or unreadable"
            found = violations_in(allowance.path, text)
            remaining, count = exempt_lines(allowance.path, found)
            assert count, (
                f"{allowance.path} is allowlisted but no longer contains any "
                "banned shape; remove the allowance"
            )
            assert not remaining, (
                f"{allowance.path} has {len(remaining)} unexempted hit(s) -- it "
                "has outgrown its ceiling"
            )

    def test_only_known_files_are_exempt(self) -> None:
        """A closed set: any exempt path not in :data:`ALLOWED` is a bug in the
        splitter rather than a policy."""
        assert set(ALLOWED) == {allowance.path for allowance in ALLOWANCES}


@pytest.mark.skipif(not _is_git_checkout(), reason="not a git checkout")
class TestGitIgnoresTheScratchDirectory:
    """The git-level half of S9.5. The ignore rules and the scan are one fix:
    the scan stops a *reference* shipping, this stops the directory itself
    ever being committed in the first place."""

    def test_check_ignore_says_it_is_ignored(self) -> None:
        result = _git("check-ignore", "-v", DEV_DIR)
        assert result.returncode == 0, (
            f"`git check-ignore {DEV_DIR}` exited {result.returncode}. The "
            "pattern is slashless on purpose: git does not treat a symlink as a "
            "directory, so trailing-slash variants never match it."
        )

    def test_the_ignore_pattern_is_named_in_gitignores_own_output(self) -> None:
        source = _git("check-ignore", "-v", DEV_DIR).stdout
        assert ".gitignore" in source, (
            f"unexpected ignore source {source!r} -- if the scratch directory is "
            "being hidden by a global gitignore, the fix has not landed here"
        )

    def test_the_scratch_directory_is_not_tracked(self) -> None:
        tracked = _git("ls-files", "-z").stdout.split("\0")
        assert DEV_DIR not in tracked, (
            f"{DEV_DIR} is tracked by git; removing it from the index is part "
            "of the same fix"
        )

    def test_the_scratch_directory_is_not_in_the_candidate_set(self) -> None:
        assert not [
            path for path in candidate_files() if path.split("/")[0] == DEV_DIR
        ], f"the scan would read files from inside {DEV_DIR}/; that defeats it"