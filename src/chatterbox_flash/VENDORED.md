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

The copy is **not unmodified**. It carries one recorded dependency swap, the
same one the vendored `chatterbox` tree took (PATCHES.md P-044): the new
`chatterbox/models/dsp.py` module stands in for librosa and scipy at this
package's two call sites — `tts.py` loads and resamples the reference clip with
`load_audio_at`/`resample` (imported at `tts.py:17`), and the evaluation
script's private `_resample` calls that same `resample` instead of
`scipy.signal.resample` (imported at `eval/wer_seedtts.py:46`). Both point at
the patched `chatterbox` tree rather than at a pip package, which is the same
reason this directory is vendored in the first place; the resampled audio is
conditioning at generation time and an already-generated clip under evaluation.

The swap cannot be recorded as a row in `src/PATCHES.md`: that manifest's rows
must name a file under `chatterbox/`, and its coverage walk only looks at
`src/chatterbox/`, so a `chatterbox_flash/` row would be invisible to the very
tests that check the manifest. This note is the record, and it has to stay
exact.

Re-sync by replacing `chatterbox_flash/` with the new version's files and
updating this file — but re-apply the two import swaps above, which a fresh
upstream copy will not carry. Any future local edit must be recorded here
before it ships.
