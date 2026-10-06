# Vendored `chatterbox-flash` — provenance

`chatterbox_flash/` is a vendored copy of the `chatterbox-flash` package,
0.1.0 (the `chatterbox_flash-0.1.0-py3-none-any.whl` build published on PyPI),
by Resemble AI — <https://github.com/resemble-ai/chatterbox-flash>. MIT
licensed; the upstream license text ships alongside as `LICENSE`.

It is vendored, never pip-installed: the package's own dependency block would
pull a pip `chatterbox-tts` over this node's vendored (patched) `chatterbox`
tree, while the Flash code deliberately imports `S3Gen`, `VoiceEncoder`,
`T3Cond` and `EnTokenizer` from `chatterbox` — i.e. from this node's patched
sources, bf16 patches included.

The copy is **unmodified** — there is no patch layer here (compare
`src/PATCHES.md`, which records the local edits to the vendored `chatterbox`
tree only). Re-sync by replacing `chatterbox_flash/` with the new version's
files and updating this file; any future local edit must be recorded in a
patch manifest before it ships.
