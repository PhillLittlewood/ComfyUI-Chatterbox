try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

# PATCHED (src/PATCHES.md P-001): upstream resolves the version through the
# installed distribution metadata. In a ComfyUI environment the package is
# vendored, never pip-installed, so `version()` raises PackageNotFoundError and
# `import chatterbox` fails before any node can be registered. Keep the lookup
# available for anyone who *does* install the distribution, but never let it
# break the import.
try:
    __version__ = version("chatterbox-tts")
except Exception:  # pragma: no cover - distribution metadata is optional here
    __version__ = "0.1.7"


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES