<a id="readme-top"></a>

<div align="center">

# ◇ ComfyUI-Chatterbox ◆ Revised ◆

<img src="https://github.com/user-attachments/assets/7f5524c3-7c02-46fc-a548-b4157dff17db" />


<p>

Chatterbox, refactored. **Text-to-Speech and Voice Conversion for ComfyUI** on a leaner core.
Supports: **classic, Turbo, Nano, Flash, Multilingual.**

*Less to import. No temp files. Same voices, faster to a result.*

</p>

<!-- PROJECT SHIELDS -->
[![Telegram][telegram-shield]][telegram-url]
[![X][x-shield]][x-url]

</div>

> [!NOTE]
> **ComfyUI-Chatterbox** runs a refactored Chatterbox core. The vendored model
> code reimplements its audio DSP in pure numpy/torch, bit-for-bit with the
> original (verified against librosa 1.0.0 on the trained featurization
> parameters), and drops `librosa`, `scipy`, `omegaconf` and `conformer`.
> Reference clips are handed to the model in memory, never through a temp
> `.wav`. The only package this node adds is `s3tokenizer`. Models download on
> first run into `ComfyUI/models/tts/chatterbox/`.

<details>
<summary><b>Table of Contents</b></summary>

* [Features](#features)
* [Models](#models)
* [Getting started](#getting-started)
* [Usage](#usage)
* [Good to know](#good-to-know)
* [Acknowledgments](#acknowledgments)

</details>

<p id="features" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❖ Features

| Capability | Detail |
| :--- | :--- |
| **One TTS node, every pack** | classic, Turbo 350M, Nano 110M and Flash are routed automatically from the pack's contents |
| **Multilingual TTS** | 23 languages, v1 / v2 / v3 weights, plus six Single Language Pack finetunes (zh-cmn, es-mx, es-es, pt-br, pt-pt, hi) |
| **Voice cloning** | any `.wav` / `.mp3` as the voice prompt, passed to the model in memory |
| **Voice conversion** | re-voice a source recording into a target timbre |
| **Multi-speaker dialog** | up to four cloned voices in one output, with SRT subtitle alignment |
| **Lean core** | DSP in pure numpy/torch, bit-exact with the original; `s3tokenizer` is the only dependency |
| **Devices** | CUDA, ROCm, XPU, MPS and CPU, under ComfyUI's VRAM manager |

<p id="models" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌸ Models

Downloaded on first run into `ComfyUI/models/tts/chatterbox/` (~2.5 GB each).
Shared weights (`ve.safetensors`, `s3gen*`) are stored once and reused across
packs.

| Family | Members | Notes |
| :--- | :--- | :--- |
| **Classic** | `chatterbox` | English TTS, built-in reference voice in `conds.pt` |
| **Turbo** | `chatterbox-turbo`, `chatterbox-turbo-nano` | Paralinguistic tags, top-k sampler; 350M / 110M |
| **Flash** | `chatterbox-flash` | Block-diffusion sampler; a reference clip is required |
| **Multilingual** | `chatterbox-mtl` | 23 languages, v1 / v2 / v3 |
| **Single Language Packs** | zh-cmn, es-mx-latam, es-es, pt-br, pt-pt, hi | MTL node only |

> [!TIP]
> Picks appear in the node's model dropdown; nothing to configure by hand.
> Ready-made graphs for every family live in [`example_workflows/`](./example_workflows).

<p id="getting-started" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ⇩ Getting started

**ComfyUI Manager** — `Manager` → `Install Custom Nodes` → search
`ComfyUI-Chatterbox` → `Install` → restart.

<details>
<summary><b>Manual installation</b></summary>

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/wildminder/ComfyUI-Chatterbox.git
pip install -r ComfyUI-Chatterbox/requirements.txt
```

Restart ComfyUI. Nodes appear under `WMNodes/sound/*`.

| Requirement | Version |
| :--- | :--- |
| Python | `>=3.10` (tested through 3.13) |
| ComfyUI | `>=0.37.0` (V3 node API) |
| Added dependency | `s3tokenizer` only, a pure-Python wheel |

</details>

Optional extras, installed as `pip install ComfyUI-Chatterbox[extra]`:
`watermarking` (resemble-perth), `multilingual` (pykakasi, spacy-pkuseg) and
`turbo` (pyloudnorm). Nothing else is required; the rest of the stack comes
from ComfyUI itself.

<p id="usage" align="center">◆◇◆◇◆◇◆◇◆◇◆</p>

## ❯ Usage

1. Add a node from `WMNodes/sound/*`.
2. Pick the model; connect an `AUDIO` input to clone a voice.
3. Queue the prompt.

| Node | Menu name | Does |
| :--- | :--- | :--- |
| `ChatterboxTTS` | Chatterbox TTS 📢 | Speech from text, optional voice cloning. One node for every TTS pack. |
| `ChatterboxMTLTTS` | Chatterbox Multilingual TTS 🌍 | Speech in 23 languages, plus the Single Language Packs. |
| `ChatterboxVC` | Chatterbox Voice Conversion 🗣️ | Re-voices a source recording into a target voice. |
| `ChatterboxDialog` | Chatterbox Multi-Speaker Dialog 🎭 | Multi-speaker conversation, up to four cloned voices. |
| `ChatterboxSRT` | Chatterbox SRT / Subtitles 📝 | Speaks a timed SRT script and returns adjusted subtitles. |

Every widget is documented in its node tooltip. Load a graph from
[`example_workflows/`](./example_workflows) to see the nodes wired up.

When a node fails at execution — a missing reference clip, a pack with no
conversion model — the message shows up as a toast in the web UI, naming the
node and the error. Add `Chatterbox.ErrorToasts` to the `Comfy.Extension.Disabled`
list in ComfyUI's settings to turn the toasts off.

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="good-to-know" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ⌗ Good to know

| | |
| :--- | :--- |
| **No version pins** | ComfyUI Manager resolves the stack; a pin could force a downgrade of the host's torch build. |
| **Offline** | Packs are fetched by a first-run download and a local pack is never re-downloaded afterwards. |
| **VRAM** | Weights load under ComfyUI's manager and offload between runs; ~2.5 GB per pack on disk as safetensors. |
| **Layout** | Node code lives in `nodes/`, the vendored Chatterbox core in `src/chatterbox/`. |
| **Upgrading from 1.x** | 2.0.0 moved packs from `models/chatterbox_tts/` to `models/tts/chatterbox/`. No compatibility shim is provided, so delete the old directory by hand. Widget order did not change and node ids did not change, so saved workflows still load. |
| **Changelog** | [CHANGELOG.md](./CHANGELOG.md) |

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>

<p id="acknowledgments" align="center">◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆</p>

## ❖ Acknowledgments

*   [Resemble AI](https://www.resemble.ai/chatterbox/) for the Chatterbox models
*   [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) for ComfyUI

<p align="right"><a href="#readme-top">⟔ ▲ ⟓ back to top</a></p>


<!-- BADGE DEFINITIONS -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/awesome-ltx2.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/awesome-ltx2/stargazers
[telegram-shield]: https://img.shields.io/badge/TokenDiff-26A5E4?style=for-the-badge&logo=telegram&logoColor=white
[telegram-url]: https://t.me/TokenDiff
[x-shield]: https://img.shields.io/badge/wildmindai-000000?style=for-the-badge&logo=x&logoColor=white
[x-url]: https://x.com/wildmindai

[contributors-shield]: https://img.shields.io/github/contributors/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[contributors-url]: https://github.com/wildminder/ComfyUI-Chatterbox/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-Chatterbox/network/members
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-Chatterbox/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-Chatterbox.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-Chatterbox/issues
