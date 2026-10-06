"""The vendored ``chatterbox`` tree must match what was actually copied.

The expected file set and the per-file SHA-256 digests are **embedded in this
file**. That is deliberate: reading an expected list from a path outside the
repository would make the assertion depend on a machine-specific checkout, and
would mean nothing shipped could verify what shipped. If a future re-sync
changes the tree, this test fails and the new digests are regenerated from what
was really copied -- which is a reviewable diff, not a silent acceptance.

Also covered here: the dead ``alignment_stream_analyzer`` module is gone and
nothing references it any more.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re

import pytest

from conftest import ROOT_DIR

VENDOR_DIR = os.path.join(ROOT_DIR, "src", "chatterbox")

#: The vendored tree, exactly as re-synced from upstream v0.1.7 plus the patch
#: layer recorded in ``src/PATCHES.md``. 51 files.
EXPECTED_FILE_HASHES: dict[str, str] = {
    "chatterbox/__init__.py": "fd8c522e8a260eb116cbf6a14d64b25b755daa4a2102c21419ec541cd26568e0",
    "chatterbox/models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "chatterbox/models/s3gen/__init__.py": "cdfafe103f5757b42bd813a1553eb481d9ea631eba34c42129399974e14a23cc",
    "chatterbox/models/s3gen/configs.py": "d5af249710db6d8bb0a3e69b03835e4e2ef20fc135d84289c13e14139868a8a0",
    "chatterbox/models/s3gen/const.py": "4e330bdc6b632afb12edb845818ffe2bf7f13441dee4d69bd74014093985fa6b",
    "chatterbox/models/s3gen/decoder.py": "ce0c531bafea8c38658b98ab8c789f14e6a23a3ece99da08631765599b0309d9",
    "chatterbox/models/s3gen/f0_predictor.py": "829f58100879516a1eea93b09c62afcf16c452795a90e76e1bb700e2db606d3f",
    "chatterbox/models/s3gen/flow.py": "2f94fc0f0e66c74e2a1f86ec3cacca08b3193ab198c711dd5d936bf3b2493157",
    "chatterbox/models/s3gen/flow_matching.py": "f5a0d68d1a00c59e3449bbe6984647eea497619405e39ff4578715bf0cae8e05",
    "chatterbox/models/s3gen/hifigan.py": "0f9adcf24117e504027a515ffa1bf2a42d5f851f4a1a79665af24f39f8660a5b",
    "chatterbox/models/s3gen/matcha/_diffusers_compat.py": "d5a6c233b241ec4e77ba79b7c41f52d0f285fe12e163be90145449fae68d24d7",
    "chatterbox/models/s3gen/matcha/decoder.py": "00a4a0c5d0714d073bf8e6be4e50650147b34ef3204907f5f35f708698093abf",
    "chatterbox/models/s3gen/matcha/flow_matching.py": "f3cdc60a786a722ec1b3bf9b12b566452a41fbe563636b71ed87b422631f36d9",
    "chatterbox/models/s3gen/matcha/text_encoder.py": "75c1e699a344866daee6edde583f60976bf90060865b23518560170e2f165c98",
    "chatterbox/models/s3gen/matcha/transformer.py": "e566d927dc7d597528628ddc167af95ad5c2814b9bec52cd67e2f437e999b69e",
    "chatterbox/models/s3gen/s3gen.py": "1bc22c8106f430875410b040c8e83ea5d1dd10e94cd86f51c9fc8152dd0509e4",
    "chatterbox/models/s3gen/transformer/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "chatterbox/models/s3gen/transformer/activation.py": "741335b4a3a3242d27c6bd471b346efc5164864043a3cf3be1ca692bfee08103",
    "chatterbox/models/s3gen/transformer/attention.py": "7ea0562b3e79cb008b7a7407826c2a09cf28f61f55defc728401126080234ab6",
    "chatterbox/models/s3gen/transformer/convolution.py": "3e85608add700e0a86a0ee472384e2db511e8b223c87d936dc4ac3b009d4b06a",
    "chatterbox/models/s3gen/transformer/embedding.py": "c6e501a052253decd26e6fa185f29d4f09890d0861b977d9b98874b8fdc03234",
    "chatterbox/models/s3gen/transformer/encoder_layer.py": "39cab20e9a22c4640c08839b9cc96c6fa10807dbfacebac41fc859d050af3ba5",
    "chatterbox/models/s3gen/transformer/positionwise_feed_forward.py": "d345af9bc7e088afd525a58adef627a6cc3b316865c11fa23c61760024901217",
    "chatterbox/models/s3gen/transformer/subsampling.py": "345ef01bf1ebca1808532f506653aadf6d35ead8205ee10716c1f0ed68e1e5a3",
    "chatterbox/models/s3gen/transformer/upsample_encoder.py": "251c9376b56b57ac0d052abf4657b83691c6331808f5f8219ebcfd2b8cd1d137",
    "chatterbox/models/s3gen/utils/class_utils.py": "b0605ec5cb1e1fbadbcc24c488ad6ca8eff3e18f5f460edd0b3027b2de1f6aed",
    "chatterbox/models/s3gen/utils/intmeanflow.py": "09f94d16a7aa8cc42ee2aff09d69fcd3ea503f1a5e69fca03d3f0786ef172b05",
    "chatterbox/models/s3gen/utils/mask.py": "0f4a0779b3ee6b70d209b630c4798cee99165560010afe3e0b275dd2105a4209",
    "chatterbox/models/s3gen/utils/mel.py": "e603d99fa827eb84b8bd3df3aef5e1e567b187ab06d02ba440e2afa0eac502b2",
    "chatterbox/models/s3gen/xvector.py": "13d6b473ce7a066c2b6ddd14893e7834e9a5705cc35383adf25c5b1947125bb7",
    "chatterbox/models/s3tokenizer/__init__.py": "fa602db9e952f86943274bacd0c5b126415938e268536b8dc551aeb0aa924bc1",
    "chatterbox/models/s3tokenizer/s3tokenizer.py": "8da30a6d978b5071761899d913b31f1fe9ae766a07c316a970b529cfcd0f0aa4",
    "chatterbox/models/t3/__init__.py": "9d4af45d74fc3c0ebef3e875d597aee3dff31e22d585385dc9567427a638e442",
    "chatterbox/models/t3/inference/t3_hf_backend.py": "98ed48dfd7e671d8953a0ea13e003ee16aaf39e8e3228799076ef08eaadc3403",
    "chatterbox/models/t3/llama_configs.py": "ab1bb91f7be9732e210b2e3e08aab3a71d31bd63aa5bdfc68bbb8fbb92b37a0c",
    "chatterbox/models/t3/modules/cond_enc.py": "d8ba6f9e8bd6b7cf0a4e0a33e62b9d6329253c892401d9cc89a331c8abe62f72",
    "chatterbox/models/t3/modules/learned_pos_emb.py": "0dd028fe53d845e71c70306175ac9e6e4d2b7ee238c0a5aa3497eed6e5e74e8a",
    "chatterbox/models/t3/modules/perceiver.py": "6790e762ba350bdc5c9a897aa13ef4f0b5639edb9cc24dc3c477917e9c1dc774",
    "chatterbox/models/t3/modules/t3_config.py": "2a5f8a336d9839cedf1c834c020ddc8980ca63c6e4c654bf4efd894b91dcafef",
    "chatterbox/models/t3/t3.py": "c8baddd6ce80b30a6cf54b1627d56fdd73a8c020bf8e73de653ae28a61fd7c9e",
    "chatterbox/models/tokenizers/__init__.py": "6759fb8b4f6d50b8e4ab8ddd0e39d44e1ca7196544f93311b385a7eeea6c38c2",
    "chatterbox/models/tokenizers/tokenizer.py": "5fe52ef58684b2756ab0f0106d62de335311da04a493b637d7d093d1870c65cf",
    "chatterbox/models/utils.py": "e35c9893b029d44e7caeb288cf4b6c836aa3b916e7529dc3a1444adfd34ab054",
    "chatterbox/models/voice_encoder/__init__.py": "a3326161831fee10ffbfa5f5eb5dbd0e43172a92f342fb645ac5db8e0d5bcda7",
    "chatterbox/models/voice_encoder/config.py": "e64be6d97b000392427a0b243dd41741888d767b96a79961a63039ffd1b2b95c",
    "chatterbox/models/voice_encoder/melspec.py": "d3d61bf55a5a3aead1a436c3d99fc0ec822c3bced6fc1b932e83605d3775bc9f",
    "chatterbox/models/voice_encoder/voice_encoder.py": "9583825f6b7b5c9cf7ec93c65172fdcb1a0622d8196db65b6a966e02e4f09fc9",
    "chatterbox/mtl_tts.py": "31f882a5333d66a0f0246023835b1928b1d347ca1b3f5997676554465c23a12f",
    "chatterbox/tts.py": "d05eaa7eda086b3c72c353a230d9404621e0482c560c87599b04420a0c9fd83b",
    "chatterbox/tts_turbo.py": "2a47a2f160de6389cc42933fe0bc46e37e0362925c0ea6b7c49a51fceba64666",
    "chatterbox/vc.py": "8a8ac8e30e05ffc3847f04231ff034f72fd9058e600f12f199616b4dd32e2f4d",
}

#: Files added by this re-sync; a regression here means someone restored the
#: previous v0.1.2 tree.
NEW_IN_THIS_SYNC = (
    "chatterbox/mtl_tts.py",
    "chatterbox/tts_turbo.py",
    "chatterbox/models/s3gen/utils/intmeanflow.py",
)

DEAD_MODULE = "chatterbox/models/t3/inference/alignment_stream_analyzer.py"


def vendor_files() -> list[str]:
    """Every vendored source file on disk, as ``chatterbox/``-rooted paths."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(VENDOR_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, name), VENDOR_DIR)
                found.append("chatterbox/" + rel.replace(os.sep, "/"))
    return sorted(found)


def sha256(rel_file: str) -> str:
    path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class TestVendorFileSet:
    def test_expected_count_is_fifty_one(self) -> None:
        assert len(EXPECTED_FILE_HASHES) == 51

    def test_vendor_dir_exists(self) -> None:
        assert os.path.isdir(VENDOR_DIR), f"missing vendored tree at {VENDOR_DIR}"

    def test_exact_file_set(self) -> None:
        on_disk = set(vendor_files())
        expected = set(EXPECTED_FILE_HASHES)
        assert on_disk == expected, (
            f"added={sorted(on_disk - expected)} removed={sorted(expected - on_disk)}"
        )

    def test_no_unexpected_files_of_any_kind(self) -> None:
        """Catches stray ``.pyc``, editor backups or hand-added modules."""
        strays: list[str] = []
        for dirpath, dirnames, filenames in os.walk(VENDOR_DIR):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                path = os.path.join(dirpath, name)
                rel = "chatterbox/" + os.path.relpath(path, VENDOR_DIR).replace(os.sep, "/")
                if rel not in EXPECTED_FILE_HASHES:
                    strays.append(rel)
        assert not strays, f"unexpected files in the vendor tree: {sorted(strays)}"

    def test_new_files_present(self) -> None:
        missing = [rel for rel in NEW_IN_THIS_SYNC if not os.path.isfile(
            os.path.join(VENDOR_DIR, rel[len("chatterbox/"):].replace("/", os.sep))
        )]
        assert not missing, f"expected new upstream files missing: {missing}"

    @pytest.mark.parametrize("rel_file", sorted(EXPECTED_FILE_HASHES))
    def test_file_hash(self, rel_file: str) -> None:
        assert sha256(rel_file) == EXPECTED_FILE_HASHES[rel_file], (
            f"{rel_file} does not match the vendored snapshot; the patch layer "
            f"in src/PATCHES.md is the contract for any intended change"
        )

    def test_every_file_parses(self) -> None:
        """The harness mocks the vendor, so syntax errors would otherwise hide."""
        broken: list[str] = []
        for rel_file in vendor_files():
            path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
            try:
                ast.parse(open(path, "r", encoding="utf-8").read(), filename=path)
            except SyntaxError as exc:
                broken.append(f"{rel_file}:{exc.lineno}: {exc.msg}")
        assert not broken, f"vendored files that do not parse: {broken}"


class TestDeadCodeRemoved:
    """The old alignment-stream analyzer was ours, not upstream's."""

    def test_analyzer_module_is_gone(self) -> None:
        path = os.path.join(VENDOR_DIR, DEAD_MODULE[len("chatterbox/"):].replace("/", os.sep))
        assert not os.path.exists(path), (
            f"{DEAD_MODULE} still exists; upstream dropped it and nothing uses it"
        )

    def test_nothing_references_the_analyzer(self) -> None:
        offenders: list[str] = []
        for rel_file in vendor_files():
            path = os.path.join(VENDOR_DIR, rel_file[len("chatterbox/"):].replace("/", os.sep))
            source = open(path, "r", encoding="utf-8").read()
            if re.search(r"alignment_stream_analyzer|AlignmentStreamAnalyzer", source):
                offenders.append(rel_file)
        assert not offenders, f"dead analyzer still referenced in: {offenders}"


class TestVendorTreeIsInertDuringCollection:
    def test_vendor_is_not_imported_for_real(self) -> None:
        import sys
        from unittest.mock import MagicMock

        module = sys.modules.get("chatterbox")
        assert module is not None, "the harness should have registered a stand-in"
        assert isinstance(module, MagicMock), (
            "the real vendor package was imported; collection must never load "
            "perth, diffusers or the model stack"
        )


class TestHarnessCoversTheNewTree:
    """``conftest.py`` derives its mock set from the tree; pin the result.

    The walk in ``conftest.vendored_module_names`` is what keeps the harness
    correct across a re-sync. If it ever stops covering a module, that module
    becomes importable for real during collection and the optional dependencies
    come back. These assertions fail before that happens.
    """

    NEW_IN_THIS_SYNC = (
        "chatterbox.mtl_tts",
        "chatterbox.tts_turbo",
        "chatterbox.models.s3gen.utils.intmeanflow",
    )

    REMOVED_IN_THIS_SYNC = (
        "chatterbox.models.t3.inference.alignment_stream_analyzer",
    )

    def _mocked(self) -> set[str]:
        import conftest

        return set(conftest.VENDORED_MODULES)

    @pytest.mark.parametrize("name", NEW_IN_THIS_SYNC)
    def test_new_module_is_mocked(self, name: str) -> None:
        assert name in self._mocked(), f"{name} is not mocked by conftest.py"

    @pytest.mark.parametrize("name", REMOVED_IN_THIS_SYNC)
    def test_removed_module_is_not_mocked(self, name: str) -> None:
        """A stale mock for a deleted module would hide a real regression."""
        assert name not in self._mocked(), (
            f"{name} is gone from the tree but still registered as a mock"
        )

    def test_every_vendor_module_is_mocked(self) -> None:
        import conftest

        on_disk = {
            "chatterbox." + rel[len("chatterbox/"):-3].replace("/", ".")
            for rel in vendor_files()
            if not rel.endswith("__init__.py")
        }
        assert on_disk, "no vendored modules found; the walk itself is broken"
        unmocked = sorted(on_disk - set(conftest.VENDORED_MODULES))
        assert not unmocked, f"vendored modules not mocked by conftest.py: {unmocked}"

    def test_mock_set_has_no_stray_names(self) -> None:
        import conftest

        strays = sorted(
            name
            for name in conftest.VENDORED_MODULES
            if not (name == "chatterbox" or name.startswith("chatterbox."))
        )
        assert not strays, f"conftest.VENDORED_MODULES carries non-vendor names: {strays}"


class TestNoMachinePathsCommitted:
    """Nothing that ships may name a path outside the repository.

    The env-gated e2e test in particular must receive its checkpoint location
    through the environment; a constant in the test file would make the file
    unrunnable on every other machine and would leak a local layout.
    """

    SHIPPED = (
        "src/PATCHES.md",
        "tests/test_e2e_smoke.py",
        "conftest.py",
    )

    WINDOWS_DRIVE = re.compile(r"[A-Za-z]:[\\/]")

    @pytest.mark.parametrize("rel_file", SHIPPED)
    def test_no_absolute_path(self, rel_file: str) -> None:
        path = os.path.join(ROOT_DIR, rel_file.replace("/", os.sep))
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        match = self.WINDOWS_DRIVE.search(source)
        assert match is None, f"{rel_file} contains an absolute path: {match.group(0) if match else ''}"

    @pytest.mark.parametrize("rel_file", SHIPPED)
    def test_no_user_home_path(self, rel_file: str) -> None:
        path = os.path.join(ROOT_DIR, rel_file.replace("/", os.sep))
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "/home/" not in source and "/Users/" not in source, (
            f"{rel_file} names a home directory"
        )

    def test_e2e_gating_is_env_only(self) -> None:
        """The checkpoint location must come from the environment, not a constant."""
        source = open(
            os.path.join(ROOT_DIR, "tests", "test_e2e_smoke.py"), "r", encoding="utf-8"
        ).read()
        for var in ("RUN_CHATTERBOX_E2E", "CHATTERBOX_MODEL_DIR", "CHATTERBOX_REF_WAV"):
            assert f'"{var}"' in source, f"the e2e test does not read {var} from the environment"