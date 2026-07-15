# Video comprehension — model & technique reference (forensic survey, mid-2026)

Companion to `video_understanding.md`. Everything here is free-to-download
(open-weights) or free/free-tier (hosted), so a high-accuracy pipeline can be
built at zero or near-zero cost. Fast-moving items are flagged ⚠️; verify
licenses/limits on the source before commercial deployment.

---

## A. Open-weights vision-language models (self-hostable, free)

VRAM rule of thumb (fp16): ≈2 GB per 1B params for weights + vision encoder +
a KV cache that grows with (#frames × tokens/frame). 4-bit ≈ 0.5–0.7 GB/1B, but
long-video KV cache can still dominate — budget headroom.

| Model family | Sizes | License | Native video? | VRAM (fp16 → 4-bit) | Best for |
|---|---|---|---|---|---|
| **Qwen3-VL** ⚠️ | 2B,4B,8B,32B dense; 30B-A3B,235B-A22B MoE (Instruct+Thinking, FP8) | Apache-2.0 | Yes — hours-long, native 256K→1M ctx, second-level indexing | 8B ~18→7 GB; 32B ~65→20 GB | **Best open temporal reasoner + long video** |
| **Qwen2.5-VL** | 3B,7B,32B,72B | 7B & 32B Apache-2.0; 3B non-commercial; 72B Qwen license (<100M MAU) | Yes — dynamic-FPS, absolute-time mRoPE, grounding | 7B ~17→6-8 GB; 32B ~65→20 GB; 72B ~145→40-48 GB | Temporal grounding, licensable all-rounder |
| **InternVL3 / 3.5** ⚠️ | 1B–78B; 3.5 adds MoE 30B-A3B, 241B-A28B, "Flash" | Apache-2.0 (backbone caveats) | Yes | 8B ~18→7 GB; 38B ~76→22 GB | High-accuracy video QA (Video-MME 38B ≈72.7/75.0) |
| **VideoLLaMA3** | 2B, 7B (+VideoRefer) | Apache-2.0 | Yes — specialist | 7B ~16→7 GB; 2B ~5 GB | Efficient temporal reasoning + object grounding |
| **LLaVA-Video** | 7B, 72B | Apache-2.0 | Yes — ≤64 frames | 7B ~16→7 GB | **Dense frame captioning** (LLaVA-Video-178K) |
| **Tarsier2-7B** ⚠️ | 7B (Qwen2-VL base) | verify card | Yes — specialist | 7B ~16→7 GB | **SOTA dense video description** (beats GPT-4o/Gemini-1.5-Pro on DREAM-1K) |
| **MiniCPM-V** | 2.6 (8B), 4.0 (~4B), 4.5 (8B) ⚠️ | code Apache-2.0; weights free (research + commercial after form) | Yes — 4.5 up to 10 FPS (3D-Resampler, 96× token compression) | 8B ~8.6 GB @ Q4 | Edge / high-FPS / low VRAM |
| **LLaVA-OneVision** | 0.5B, 7B, 72B | Apache-2.0 | Yes (image+multi-image+video) | 7B ~16→7 GB | Balanced image+video generalist |
| **SmolVLM2** | 256M, 500M, 2.2B | Apache-2.0 | Yes | 2.2B 5.2 GB / 500M 1.8 GB | Ultra-low-VRAM / mobile |
| Molmo, Pixtral | 1B–124B | Apache-2.0 (mostly) | **No** (image only) | — | Image QA/OCR — **not for video** |
| Llama 4 Scout/Maverick | 109B/400B (17B active) | Llama 4 Community (<700M MAU) | early-fusion, **video-marginal** | Scout ~55 GB @Q4 | Huge context, image-heavy docs |

**Picks:** dense captioning → LLaVA-Video-7B / Tarsier2-7B (quality) or MiniCPM-V 4.5
(FPS/VRAM). Temporal + long video → Qwen3-VL → Qwen2.5-VL-32B/InternVL3-38B →
VideoLLaMA3-7B. Edge → SmolVLM2 / MiniCPM-V / Qwen3-VL-2B-4B.

Key sources: Qwen2.5-VL [2502.13923](https://arxiv.org/pdf/2502.13923) ·
Qwen3-VL [github](https://github.com/qwenlm/qwen3-vl) / [2511.21631](https://arxiv.org/pdf/2511.21631) ·
InternVL3 [2504.10479](https://arxiv.org/pdf/2504.10479) / InternVL3.5 [2508.18265](https://arxiv.org/abs/2508.18265) ·
VideoLLaMA3 [2501.13106](https://arxiv.org/html/2501.13106v1) ·
LLaVA-Video [2410.02713](https://arxiv.org/html/2410.02713v3) ·
Tarsier2 [2501.07888](https://arxiv.org/abs/2501.07888) ·
MiniCPM-V [github](https://github.com/OpenBMB/MiniCPM-V) ·
SmolVLM2 [blog](https://huggingface.co/blog/smolvlm2).

---

## B. Speech-to-text (ASR), free

| Option | Type | Notes |
|---|---|---|
| **faster-whisper** (CTranslate2) | open weights | Fastest local; low VRAM; word timestamps + VAD. Default in `asr.py`. |
| **Whisper large-v3 / distil-whisper** | open weights | Reference quality; distil ≈6× faster. CPU-capable at small sizes. |
| **Groq Whisper** | hosted free tier | No local GPU; very fast; good for batch transcription. (limits in §D) |

---

## C. Anti-hallucination & temporal-reasoning techniques (the swarm's backbone)

1. **Self-consistency** — sample K captions, keep claims recurring in ≥⌈K/2⌉;
   drop low-frequency (likely hallucinated) detail.
   [2203.11171](https://arxiv.org/abs/2203.11171) · VLM variant [2509.23236](https://arxiv.org/abs/2509.23236).
2. **Multi-agent debate / ensemble critics** — independent models critique & revise;
   uncorrelated errors cancel. [2305.14325](https://arxiv.org/abs/2305.14325) ·
   N-Critics [2310.18679](https://arxiv.org/abs/2310.18679).
3. **Chain-of-Verification** — draft → verification questions → answer them
   *independently* → rewrite. [2309.11495](https://arxiv.org/abs/2309.11495).
4. **Verifier/critic loops** — Self-Refine [2303.17651](https://arxiv.org/abs/2303.17651),
   Reflexion [2303.11366](https://arxiv.org/abs/2303.11366),
   Woodpecker post-hoc detector-grounded correction [2310.16045](https://arxiv.org/abs/2310.16045).
5. **Timestamp grounding** — force each event to carry a defensible [start,end]:
   TimeChat [2312.02051](https://arxiv.org/abs/2312.02051),
   Grounded-VideoLLM [2410.03290](https://arxiv.org/abs/2410.03290),
   VTimeLLM [2311.18445](https://arxiv.org/abs/2311.18445).
6. **VideoRAG** — retrieve relevant segments instead of cramming the whole video:
   [2502.01549](https://arxiv.org/abs/2502.01549) · [2501.05874](https://arxiv.org/abs/2501.05874).
7. **Adaptive sampling** — AKS [2502.21271](https://arxiv.org/abs/2502.21271),
   AdaRD-Key [2510.02778](https://arxiv.org/abs/2510.02778),
   16-FPS-when-needed [2503.13956](https://arxiv.org/abs/2503.13956).
8. **Detector-grounded object consistency** — Grounding DINO + SAM 2 tracking; flag
   caption objects with no detector support (CHAIR metric [1809.02156](https://arxiv.org/abs/1809.02156)).
   Caveat: grounding must be *verified*, not assumed [2406.14492](https://arxiv.org/abs/2406.14492).
9. **Contrastive decoding** (drop-in, training-free) — VCD [2311.16922](https://arxiv.org/abs/2311.16922),
   TCD / EventHallusion [2409.16597](https://arxiv.org/abs/2409.16597).
10. **Evaluate** on VideoHallucer [2406.16338](https://arxiv.org/abs/2406.16338) +
    EventHallusion; instrument CHAIR to track object hallucination over time.

**Recommended layering:** adaptive sampling → timestamp-grounded captioning →
detector-grounded object check → self-consistency voting → CoVe/Woodpecker rewrite →
VideoRAG at answer time → evaluate on VideoHallucer, add VCD/TCD if temporal
hallucination persists. This is exactly the order the AgentVerse swarm + perception
pipeline implement.

---

## D. Free / free-tier hosted APIs (forensic survey, 2026-07-15)

> **Verify before shipping.** This space churned hard in H1 2026 (Gemini Dec-2025
> quota cuts + 2.0-Flash retirement, Groq/Mistral vision deprecations, Cerebras
> catalog collapse, Together credit removal, HF pricing migration). Every number
> below should be re-checked in the provider's own console. Many official docs
> hosts were egress-blocked during research, so figures come from search-surfaced
> snippets of those same official pages + trackers.

### Headline finding
**Only three free paths accept VIDEO input:** **Google Gemini** (native mp4, the
standout), **NVIDIA NIM** (Cosmos-Reason1-7B, Nemotron Omni), and a few
**OpenRouter `:free`** Nemotron/Gemma models. Everything else is image-only or
text-only on the free tier.

| Provider | Truly free? | Image | **Video** | Best free vision IDs | Ctx | Free limits (headline) |
|---|---|---|---|---|---|---|
| **Google Gemini** | free tier of paid | ✅ | ✅ **native** | `gemini-2.5-flash`, `-flash-lite`, `-2.5-pro` | **1M** | Pro 5 RPM/100 RPD; Flash 10/250; Flash-Lite 15/1,000; 250K TPM |
| **NVIDIA NIM** | ✅ Dev Program | ✅ | ✅ Cosmos-Reason1, Nemotron Omni | `nvidia/cosmos-reason1-7b`, `meta/llama-3.2-90b-vision` | ≤1M | ~40 RPM (200 on req) |
| **OpenRouter** | ✅ `:free` | ✅ | ✅ some | `nvidia/nemotron-nano-omni:free`, `google/gemma-*:free`, Qwen3.5-VL `:free` | ≤262K | 20 RPM; **50 RPD (<$10) / 1,000 RPD (≥$10 lifetime)** |
| **Groq** | ✅ no card | ✅ (Llama 4) | ❌ | `meta-llama/llama-4-scout-17b-16e-instruct` ⚠️deprecating | 128K | ~30 RPM/1,000 RPD + **free Whisper** |
| **GitHub Models** | ✅ prototyping | ✅ | ❌ | GPT-4o, Llama-3.2-Vision, Phi-3.5-vision | **8K in/4K out cap** | High 10 RPM/50 RPD; Low ~15/150 |
| **Cloudflare Workers AI** | ✅ daily | ✅ (LLaVA) | ❌ | `@cf/meta/llama-3.2-11b-vision-instruct`, `@cf/meta/llama-4-scout...` | ≤128K | **10,000 Neurons/day** (~$0.11) |
| **Mistral** | ✅ Experiment | ✅ (OCR strong) | ❌ | `pixtral-*` ⚠️deprecated → `mistral-medium` | 128K | ~1 req/s (~2 RPM) |
| **Together AI** | ~$5 min | ✅ | ❌ | `meta-llama/Llama-Vision-Free` ⚠️time-limited | 131K | dynamic/unpublished |
| **HuggingFace** | trial credit | ✅ (routed) | ❌ | `Qwen/Qwen2.5-VL-*-Instruct` | model | **$0.10/mo** ($2 PRO) |
| **Cerebras** | ✅ no card | ❌ | ❌ | none (text-only) | 8K free | 5–30 RPM / **1M tokens/day** |

### Notes that matter for a comprehension pipeline
- **Gemini free tier + video:** samples video at ~1 fps with per-second timestamps;
  1M-ctx ≈ 1 hr of video (2M ≈ 2 hr). Inline <~20 MB, else the Files API (≤100 MB+).
  RPD/RPM bind long before TPM. **`gemini-2.0-flash` retired 2026-03-03 — use 2.5.**
  [rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits) ·
  [video-understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- **Data-use caveat (the real cost of "free"):** Gemini **free tier trains on your
  prompts + human reviewers may read them** (paid does not; EEA/CH/UK get paid terms
  even when free). OpenRouter `:free` **requires enabling prompt training/logging**.
  NVIDIA NIM free endpoints may log/train. **→ Do not send sensitive video to free
  tiers;** self-host (§A) for confidential material.
- **Free ASR:** **Groq Whisper** (`whisper-large-v3` / `-turbo`) ~2,000 RPD +
  7,200 audio-sec/hr, ≤100 MB [speech-to-text](https://console.groq.com/docs/speech-to-text);
  **Cloudflare** `@cf/openai/whisper-large-v3-turbo` (~214 audio-min/day free);
  else self-host **faster-whisper / distil-whisper** (free forever, `asr.py` default).

### Recommended free stacks
- **Zero-infra, best free accuracy:** Gemini 2.5 Flash (primary video captioner +
  answerer) × OpenRouter `nemotron-nano-omni:free` or NVIDIA Cosmos-Reason1
  (independent 2nd-opinion captioner/critic) × Groq Whisper (ASR) ×
  Cerebras/Groq text models (verifier debate — 1M tokens/day is ample).
- **Confidential / unlimited:** self-host Qwen3-VL or LLaVA-Video-7B + faster-whisper
  (§A/§B). No data leaves your machine; the only cost is one GPU.
- **Independence rule:** always cross-check with backends from *different* providers/
  weights — correlated errors are what defeat a single-model pipeline.
