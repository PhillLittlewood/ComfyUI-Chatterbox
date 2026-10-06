# Vendored `chatterbox` — patch manifest

The `src/chatterbox/` tree is a vendored copy of upstream `resemble-ai/chatterbox`
(MIT). It is re-synced by **taking upstream first, then re-applying this layer** —
never the reverse. This file is the contract: every entry below is a local edit to
vendored source, and `tests/test_vendor_patch_manifest.py` fails if an entry's
symbol is missing from the tree, or if the file leaks a path outside the repository.

Provenance is recorded by rationale, not by pointer: nothing here may name the
scratch directory the upstream checkout was read from.

## Categories

| Category | Meaning |
| --- | --- |
| `ComfyUI-API patch` | The ComfyUI node surface — an argument, progress report or option the node passes that upstream does not have. |
| `compat patch` | Upstream regressed something, or assumes an environment ours does not provide. |
| `feature addition` | New capability with no upstream counterpart. |

## Patch manifest

Paths are relative to `src/`. Symbols are module-level names, `Class.attr` pairs,
or function names, exactly as they appear in the source.

Rows are grouped per file, and a later batch's row is appended to the run it
belongs to rather than at the end of the table, so the IDs do not read in
ascending order.

| ID | File | Symbol | Category | Rationale |
| --- | --- | --- | --- | --- |
| P-001 | `chatterbox/__init__.py` | `__version__` | compat patch | Upstream calls `importlib.metadata.version("chatterbox-tts")` at import time. The package is vendored, never pip-installed, so the lookup raises `PackageNotFoundError` and `import chatterbox` dies before any node registers. The lookup is kept but its failure is non-fatal. |
| P-002 | `chatterbox/tts.py` | `_pad_wav_to_40ms_multiple` | compat patch | The mel spectrogram hops every 20 ms and the S3 tokenizer emits one token every 40 ms. A reference clip that is not an exact multiple of 40 ms produces `mel_len != 2 * token_len`, which `embed_ref` only repairs by dropping tokens. Pad before resampling instead. |
| P-003 | `chatterbox/tts.py` | `ChatterboxTTS.from_local` | compat patch | `device in ["cpu", "mps"]` is False for every `torch.device`, so a CPU load selected `map_location=None` and tried to restore CUDA-saved tensors. Compare `str(device)`. |
| P-004 | `chatterbox/tts.py` | `ChatterboxTTS.prepare_conditionals` | compat patch | Applies P-002: the loaded clip is round-tripped through `torch` so it can be padded onto the token grid before being resampled to 16 kHz. |
| P-005 | `chatterbox/tts.py` | `ChatterboxTTS.generate` | ComfyUI-API patch | Accepts `pbar`, `max_new_tokens` and `flow_cfg_scale`, and forwards them to `T3.inference` and `S3Gen.inference`. The node exposes all three as widgets; upstream hardcodes `max_new_tokens=1000` and takes no progress report. |
| P-023 | `chatterbox/tts.py` | `_apply_watermark` | feature addition | Upstream hard-imports `perth` and constructs `perth.PerthImplicitWatermarker()` in `__init__`, so `import chatterbox.tts` fails outright without an optional watermarking library. The import is now guarded, construction yields `None` when the library is absent, and `generate()` takes `watermark: bool = True` — the opt-out upstream has no equivalent of. Default `True` keeps upstream behaviour exactly; `False` skips `apply_watermark()`; `True` without the library warns once on the named `chatterbox.watermark` logger and returns the tensor unchanged. |
| P-027 | `chatterbox/tts.py` | `_require_checkpoint_files` | compat patch | `ChatterboxTTS.from_local` opens with a completeness check over the four files a classic pack cannot be built without, raising `FileNotFoundError` that names the missing files and then the full set. Upstream reaches its first `load_file` on a path that does not exist and dies with `FileNotFoundError: .../ve.safetensors` — one file named, the other three left to guesswork, which is exactly the failure a community finetune or a half-cancelled download produces. The helper is duplicated per entry point rather than shared through a new vendored module, so each file stays independently vendorable and the vendored file set stays at 51; P-028…P-030 are the same patch against the other three `from_local` implementations, with the file set each of those actually needs. This is also what lets an arbitrary local directory — Viterbox, Egyptian Arabic, anything a user drops under `models/tts/chatterbox` — be an ordinary pack rather than a special case: it clears exactly the same bar as an official one, and is refused in exactly the same words when it does not. |
| P-006 | `chatterbox/vc.py` | `_pad_wav_to_40ms_multiple` | compat patch | Same token-grid padding as P-002, for the voice-conversion reference clip. |
| P-007 | `chatterbox/vc.py` | `ChatterboxVC.from_local` | compat patch | Same `str(device)` comparison as P-003. |
| P-008 | `chatterbox/vc.py` | `ChatterboxVC.set_target_voice` | compat patch | Applies P-006: pads the reference clip onto the token grid before truncating to `DEC_COND_LEN`. |
| P-009 | `chatterbox/vc.py` | `ChatterboxVC.generate` | ComfyUI-API patch | Accepts `n_timesteps`, `pbar`, `temperature` and `flow_cfg_scale` and forwards them to `S3Gen.inference`. Upstream exposes only `audio` and `target_voice_path`; the node drives all four as widgets. |
| P-024 | `chatterbox/vc.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the voice-conversion entry point, including the `watermark` kwarg on `generate()`. |
| P-028 | `chatterbox/vc.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxVC.from_local`, which needs only `s3gen.safetensors` — a conversion-only pack loads no T3 weights, so the message names a one-file set rather than a four-file one. |
| P-010 | `chatterbox/models/t3/modules/perceiver.py` | `AttentionQKV.flash_attention` | compat patch | Upstream calls `torch.backends.cuda.sdp_kernel`, deprecated in torch 2.2 and removed later. Replaced with `torch.nn.attention.sdpa_kernel([FLASH, EFFICIENT, MATH])`, with the old call kept as the torch < 2.2 fallback. |
| P-011 | `chatterbox/models/s3gen/matcha/transformer.py` | `FeedForward` | compat patch | `LoRACompatibleLinear` is replaced by `nn.Linear` at both construction sites, and `diffusers.models.lora` is no longer imported. Modern diffusers no longer exports the class, and the LoRA scaling it added is unused by this inference-only model. |
| P-021 | `chatterbox/models/s3gen/matcha/transformer.py` | `BasicTransformerBlock` | compat patch | The four activations, the two adaptive norms, `Attention` and `maybe_allow_in_graph` are imported from `._diffusers_compat` instead of `diffusers.models.attention`, `diffusers.models.attention_processor` and `diffusers.utils.torch_utils`. Completes P-011: the file no longer imports `diffusers` at all. |
| P-020 | `chatterbox/models/s3gen/matcha/_diffusers_compat.py` | `Attention` | compat patch | New local module holding the eight `diffusers` symbols the Matcha decoder imported (`GEGLU`, `GELU`, `ApproximateGELU`, `AdaLayerNorm`, `AdaLayerNormZero`, `Attention`, `get_activation`, `maybe_allow_in_graph`), copied from `diffusers` (Apache-2.0, notice retained) and reduced to the parameters `models/s3gen/matcha/` actually reaches. `diffusers` is a heavyweight training-only library this inference-only decoder pulled in, and the build otherwise installed cannot be imported against `huggingface_hub` 1.x. The module docstring lists every dropped path, the upstream defects kept verbatim because the code that would hit them is unreachable, and the one deliberate improvement (`scale_qk` is honoured via SDPA's `scale=`). |
| P-022 | `chatterbox/models/s3gen/matcha/decoder.py` | `Decoder` | compat patch | `get_activation` is imported from `._diffusers_compat` instead of `diffusers.models.activations`. |
| P-012 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.lock` | feature addition | A TensorRT estimator is a stateful single-stream object; two workflows sharing one model instance can enter it concurrently. Guarded by a `threading.Lock`. |
| P-013 | `chatterbox/models/s3gen/flow_matching.py` | `CausalConditionalCFM.forward` | ComfyUI-API patch | Accepts `pbar` and `flow_cfg_scale`. `pbar` is forwarded to the Euler solver; `flow_cfg_scale` overrides the baked-in `inference_cfg_rate` for the duration of the call. Also applied to `ConditionalCFM.forward` and `basic_euler`. |
| P-014 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.solve_euler` | ComfyUI-API patch | Accepts `pbar` and advances it once per Euler step, so the node reports progress across the mel-generation loop instead of appearing hung. |
| P-015 | `chatterbox/models/s3gen/flow_matching.py` | `ConditionalCFM.forward_estimator` | feature addition | Lets a TensorRT `IExecutionContext`-style object stand in for the torch estimator module. The TRT path rebinds input shapes, calls `execute_v2` with raw device pointers under :data:`P-012`'s lock, and returns the `x` buffer the engine wrote into. |
| P-016 | `chatterbox/models/s3gen/flow.py` | `CausalMaskedDiffWithXvec.inference` | ComfyUI-API patch | Accepts `pbar`, `temperature` and `flow_cfg_scale` and forwards them to the decoder. Upstream's signature is fixed and offers no progress report. |
| P-017 | `chatterbox/models/s3gen/flow.py` | `CausalMaskedDiffWithXvec.fp16` | ComfyUI-API patch | Half-precision switch. The fp32 checkpoint stays resident and only the conditioning tensors (`prompt_feat`, `embedding`) are downcast; the returned mel is cast back to fp32 for the vocoder. |
| P-018 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Wav.inference` | ComfyUI-API patch | `S3Token2Mel.forward`, `S3Token2Wav.flow_inference`, `S3Token2Wav.forward` and `S3Token2Wav.inference` (exported as `S3Gen`) each accept `pbar`, `temperature` and `flow_cfg_scale` and thread them down to `CausalConditionalCFM`. Every upstream kwarg (`n_cfm_timesteps`, `noised_mels`, `skip_vocoder`, `speech_token_lens`, `drop_invalid_tokens`) is preserved. |
| P-019 | `chatterbox/models/t3/t3.py` | `T3.inference` | ComfyUI-API patch | Accepts `pbar` and advances it once per sampled token. `max_new_tokens` keeps upstream's default of `None` but now falls back to `self.hp.max_speech_tokens` instead of raising, so the node may omit it. |
| P-025 | `chatterbox/mtl_tts.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the multilingual entry point. Also applies the `str(device)` comparison of P-003 and P-007 in `ChatterboxMultilingualTTS.from_local`, which upstream still compares raw. |
| P-029 | `chatterbox/mtl_tts.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxMultilingualTTS.from_local`. Its set differs: `ve.pt`, `s3gen.pt` and `grapheme_mtl_merged_expanded_v1.json`, plus the T3 checkpoint for the selected version. That last name is not fixed, so `from_local` appends the already-resolved `t3_model` to the tuple rather than the helper hard-coding it. |
| P-026 | `chatterbox/tts_turbo.py` | `_apply_watermark` | feature addition | Same optional-`perth` patch as P-023 for the turbo entry point. Also applies the `str(device)` comparison of P-003 and P-007 in `ChatterboxTurboTTS.from_local`, which upstream still compares raw, and makes `pyloudnorm` an optional import — it is used only by `norm_loudness`, which already degrades gracefully, so the module-level hard import was making the whole entry point unimportable without it. |
| P-030 | `chatterbox/tts_turbo.py` | `_require_checkpoint_files` | compat patch | Same patch as P-027 for `ChatterboxTurboTTS.from_local`, covering Turbo and Nano together — there is no separate Nano module, so there is no separate `from_local`. The variant's T3 checkpoint (`t3_turbo_v1.safetensors` or `t3_nano_v1.safetensors`) is appended by the caller for the same reason as P-029. |

## Upstream changes adopted verbatim

These arrive with the v0.1.7 sources. No local edit, but each was a deliberate
upgrade over what this repository previously carried, and each is asserted by
`tests/test_vendor_signatures.py::TestUpstreamFixes` so a future re-sync cannot
silently revert it.

| ID | File | Symbol | Rationale |
| --- | --- | --- | --- |
| U-001 | `chatterbox/models/s3gen/s3gen.py` | `S3Token2Mel.__init__` | `CAMPPlus(memory_efficient=False)`. Activation checkpointing is a training optimization that makes `accelerate`'s DDP wrapper crash at inference. |
| U-002 | `chatterbox/models/voice_encoder/voice_encoder.py` | `VoiceEncoder.embeds_from_wavs` | `librosa.resample(..., res_type="kaiser_fast")` instead of `kaiser_best`. Same quality for voice timbre, a fraction of the time. |
| U-003 | `chatterbox/models/s3gen/const.py` | `S3GEN_SIL` | The sentinel silence token id, needed by the token-dropping path. |
| U-004 | `chatterbox/models/s3gen/utils/mask.py` | `make_pad_mask` | `lengths.long()` before comparison. An int32 length tensor compared against a bool mask silently takes the wrong branch. |
| U-005 | `chatterbox/models/t3/llama_configs.py` | `LLAMA_520M_CONFIG_DICT` | `attn_implementation="sdpa"`. `"eager"` was needed only to expose attention weights to the removed alignment analyzer. |
| U-006 | `chatterbox/models/t3/inference/t3_hf_backend.py` | `T3HuggingfaceBackend` | The dead `alignment_stream_analyzer` argument is gone upstream, along with the `models/t3/inference/alignment_stream_analyzer.py` module. |

## Known upstream leftovers

None. The two Matcha files that still reached into `diffusers` after P-011 no
longer do: P-020 vendors the eight symbols they used as
`chatterbox/models/s3gen/matcha/_diffusers_compat.py`, and P-021/P-022 point the
imports at it. `tests/test_diffusers_compat.py` asserts that no vendored file
imports `diffusers`, and that the compat module's attention is numerically equal
to a hand-written `scaled_dot_product_attention` reference.

## Repository-side changes

These are this repository's own files, not vendored ones, so they carry no patch
id above; recorded here because they are the other half of the same change.

- `modules/vendor_loader.py` (new) — `ensure_vendor_on_path()`, one idempotent
  `sys.path.append` of this repository's `src/`. `append`, never
  `insert(0, ...)`, so a pip-installed `chatterbox` keeps winning.
- `modules/__init__.py` (new, empty) — makes `modules` a real package so the
  loader can be imported without re-exports.
- `__init__.py` — dropped `sys.path.insert(0, src_dir)` (now
  `ensure_vendor_on_path()`) and dropped the `sys.modules['perth'] = perth_mock`
  stub injection, which the optional-`perth` patches P-023…P-026 make
  unnecessary. Importing the node package no longer mutates global import state.
- `modules/chatterbox_handler.py` — `chatterbox.tts` / `chatterbox.vc` are
  imported inside `load_chatterbox_models` instead of at module scope, so ComfyUI's
  node load no longer pulls librosa, transformers and the model stack in.