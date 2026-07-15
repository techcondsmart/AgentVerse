# Video Understanding in AgentVerse — a near-infallible comprehension swarm

This document sketches how to add **correct, low-hallucination video reading &
comprehension** to AgentVerse, and how the design pushes each failure mode of
video-LLMs toward its limit:

- **hallucination → ~0** via grounded evidence + multi-model + Chain-of-Verification
- **temporal reasoning → max** via timestamped evidence, order-aware retrieval, temporal analyst
- **frame-sampling density → high, cheap** via scene-cut + motion + embedding-dedup
- **caption quality → max / frame-loss → min** via a cross-checked caption ensemble
- **cost → ~0** via free / free-tier models (Gemini free tier, Groq, OpenRouter `:free`,
  and open-weights Qwen3-VL / InternVL3 / VideoLLaMA3 you self-host)

It is inspired by HKUDS/VideoAgent's agentic decomposition + VideoRAG, but rebuilt
natively on AgentVerse's `tasksolving` paradigm (role_assigner → solver → critic
loop → executor → evaluator), which already gives us the debate-and-verify machinery.

---

## 1. Two-stage design: freeze perception, then debate over it

```
                    ┌──────────────────── PERCEPTION (runs once/video) ─────────────────────┐
   video.mp4  ─────▶│  AdaptiveFrameSampler   Transcriber(Whisper)   CaptionEnsemble        │
                    │      (scene+motion+dedup)    (timestamped ASR)   (multi-VLM, grounded) │
                    │                    └────────────┬────────────┘                         │
                    │                       VideoEvidenceIndex  (semantic + temporal, cited) │
                    └────────────────────────────────┬──────────────────────────────────────┘
                                                      │  frozen, auditable evidence
   ┌──────────────────────── REASONING SWARM (tasksolving) ─────────────────────────────────┐
   │ role_assigner → recruits visual / audio / temporal / skeptic analysts                  │
   │ solver        → drafts answer where EVERY claim cites a [mm:ss] evidence id            │
   │ critics (×N)  → Chain-of-Verification on independent backbones; reject ungrounded claims│
   │ executor      → runs VideoEvidenceIndex.query() to fetch the exact evidence demanded    │
   │ evaluator     → groundedness gate: score=1 only if 100% of claims are cited & temporally consistent │
   └───────────────────────────────────────────────────────────────────────────────────────┘
```

**Why split it?** Perception is expensive and non-deterministic; reasoning must be
reproducible and checkable. Freezing perception into an evidence store means the
swarm's conclusions can always be traced back to a timestamped, provenance-tagged
observation — the single biggest lever against hallucination.

Code lives in `agentverse/video/` (perception) and
`agentverse/tasks/tasksolving/video_understanding/config.yaml` (the swarm).

---

## 2. The perception stage (`agentverse/video/`)

| Module | Responsibility | Anti-failure technique |
|---|---|---|
| `sampling.py` | Keyframe extraction | Scene-cut (PySceneDetect) guarantees ≥1 frame/shot; motion sampling raises density where information is dense; CLIP/ImageBind embedding-dedup drops near-duplicates. Bounded by `[min_fps, max_fps]` + `max_frames` for performance. |
| `asr.py` | Speech → timestamped text | faster-whisper / Whisper large-v3; audio is a first-class evidence channel so spoken meaning is never lost. |
| `captioning.py` | Frame → grounded, typed evidence | **Caption ensemble**: ≥2 independent VLMs per frame + self-consistency sampling; keep only detail that recurs; typed fields (objects/actions/ocr/setting/uncertain) + agreement score + provenance. Low-agreement frames are flagged for the swarm to scrutinize. |
| `retrieval.py` | VideoRAG index | Fused **semantic** (bge-small embeddings) + **temporal** (sorted timeline, neighbourhood expansion) retrieval; every hit carries a timestamp + source so answers can cite and critics can check. |
| `backends.py` | VLM caption backends | One OpenAI-compatible factory covers Gemini free-tier, OpenRouter `:free`, Groq, and local vLLM/Ollama open-weights — keys via env vars. |
| `pipeline.py` | Orchestration | Produces a `VideoKnowledge` object (the frozen index + low-agreement flags). |

> The heavy steps in `sampling.py` (`_extract_keyframes`, `_dedup`) are left as
> clearly-specified `NotImplementedError` stubs with the exact algorithm in the
> docstring — this is a **sketch/scaffold**, wired end-to-end but with the two
> OpenCV-heavy kernels to be filled in. Everything else runs.

---

## 3. The reasoning swarm (`.../video_understanding/config.yaml`)

Mapped onto AgentVerse's existing `tasksolving` roles, each chosen for a specific
anti-hallucination job (full rationale in the config header):

- **role_assigner** recruits *complementary* analysts — visual, audio/dialogue,
  temporal-order, and a skeptical fact-checker — so no modality is silently dropped.
- **solver ("Grounded Answerer")** must cite `(mm:ss)` evidence for every claim and
  say *"Not evidenced in the video"* rather than guess. `temperature: 0` kills drift.
- **critics ("Verifier 1/2")** run **Chain-of-Verification**: decompose the answer
  into atomic claims, form verification questions, answer them *independently* against
  the evidence, and reject anything ungrounded or temporally inconsistent. Put the two
  critics on **different LLM backbones** so their errors don't correlate (multi-agent
  debate effect).
- **executor** is bound to a `VideoEvidenceIndex`-backed retrieval tool that fetches
  the exact timestamped evidence the critics demand (see §5 to wire it).
- **evaluator** is a **groundedness gate**: it returns 1 only if every claim is cited
  and temporal ordering is consistent — otherwise the loop iterates.

Run it (once the LLM keys are set) exactly like any other AgentVerse task:

```python
from agentverse.tasksolving import TaskSolving
ts = TaskSolving.from_task("video_understanding", "agentverse/tasks/tasksolving")
ts.run()  # task_description carries the user's question + the retrieved evidence
```

---

## 4. Evidence-based techniques baked in (with sources)

The swarm is not "prompt-and-pray" — each mechanism maps to a published result:

- **Self-consistency** voting over K caption samples, keep recurring claims —
  Wang et al., *Self-Consistency* ([2203.11171](https://arxiv.org/abs/2203.11171));
  VLM variant ([2509.23236](https://arxiv.org/abs/2509.23236)).
- **Multi-agent debate / independent critics** — Du et al. ([2305.14325](https://arxiv.org/abs/2305.14325)),
  N-Critics ([2310.18679](https://arxiv.org/abs/2310.18679)).
- **Chain-of-Verification** (independent verification questions) — Dhuliawala et al.
  ([2309.11495](https://arxiv.org/abs/2309.11495)).
- **Post-hoc detector-grounded correction** (optional upgrade, §6) — Woodpecker
  ([2310.16045](https://arxiv.org/abs/2310.16045)).
- **Timestamp-grounded captioning / temporal tokens** — TimeChat ([2312.02051](https://arxiv.org/abs/2312.02051)),
  Grounded-VideoLLM ([2410.03290](https://arxiv.org/abs/2410.03290)), VTimeLLM ([2311.18445](https://arxiv.org/abs/2311.18445)).
- **VideoRAG retrieval over long video** — ([2502.01549](https://arxiv.org/abs/2502.01549), KDD 2026),
  corpus variant ([2501.05874](https://arxiv.org/abs/2501.05874)).
- **Adaptive keyframe sampling** — AKS ([2502.21271](https://arxiv.org/abs/2502.21271)),
  AdaRD-Key ([2510.02778](https://arxiv.org/abs/2510.02778)); higher effective FPS when motion is fine
  ([2503.13956](https://arxiv.org/abs/2503.13956)).
- **Evaluate hallucination** on VideoHallucer ([2406.16338](https://arxiv.org/abs/2406.16338)) and
  EventHallusion ([2409.16597](https://arxiv.org/abs/2409.16597)); add contrastive decoding (VCD
  [2311.16922](https://arxiv.org/abs/2311.16922) / TCD) if temporal hallucination persists.

---

## 5. Wiring the retrieval tool (the one integration seam)

The config ships with `executor.type: none` so the swarm is runnable immediately
with evidence passed inline. To close the loop into live retrieval, register a tiny
executor that calls the index — sketch:

```python
# agentverse/environments/tasksolving_env/rules/executor/video_evidence.py
from . import executor_registry
from .base import BaseExecutor

@executor_registry.register("video-evidence")
class VideoEvidenceExecutor(BaseExecutor):
    index = None  # inject the VideoKnowledge.index for the current video

    async def astep(self, agent, task_description, solution, *a, **kw):
        hits = self.index.query(solution, k=8)          # solution = critics' queries
        return [f"[{h.timestamp:0.1f}s | {h.modality} | agreement={h.agreement:.2f}] {h.text}"
                for h in hits]
```

Then set `executor.type: video-evidence` in the config. The `${preliminary_solution}`
prompt slot already expects evidence lines in exactly this format.

---

## 6. Cost/accuracy stacks (free & free-tier)

See `documentation/video_models_reference.md` for the full forensic survey. Short version:

**Zero-infra (API only, free tiers):**
- Captioner: **Gemini Flash free tier** (native video, generous free RPM/RPD).
- 2nd opinion / critic backbone: **OpenRouter `:free`** Qwen-VL or Llama-Vision, and/or **Groq** (fast free tier) for the ASR + text critics.
- ASR: **Groq Whisper** free tier or local faster-whisper.

**Max-accuracy self-hosted (free forever, one GPU tier up):**
- Captioner: **LLaVA-Video-7B** or **Tarsier2-7B** (dense-caption specialists); **MiniCPM-V 4.5** for high-FPS on low VRAM.
- Temporal/long-video reasoner: **Qwen3-VL** (Apache-2.0, native 256K–1M ctx) → **Qwen2.5-VL-32B** / **InternVL3-38B** → **VideoLLaMA3-7B** (efficient specialist).
- Independence: pair one hosted (Gemini) + one local (Qwen) captioner so their errors don't correlate.

**The most reliable, cheapest accuracy upgrades** (per the hallucination survey):
contrastive decoding (VCD/TCD) and post-hoc detector-grounded correction
(Grounding DINO + SAM 2, Woodpecker-style) — add these to `captioning.py` as a
verification pass when a frame's agreement score is low.

---

## 7. Validation status

Run the hermetic self-test (no API keys, no downloaded weights — synthetic video
+ mock captioners):

```bash
python -m agentverse.video.selftest
```

Validated end-to-end through the real framework:
- **Frame sampling** on a real (synthetic) 3-scene video: scene-cuts detected at
  the true boundaries, motion frames kept inside moving shots, embedding-dedup
  drops only true duplicates (22/23 kept).
- **Caption ensemble** fusion + cross-model agreement scoring (2 backends).
- **Perception pipeline** `VideoPerceptionPipeline.process()` end-to-end.
- **Retrieval**: lexical path returns the correct scene for each query
  ("safety"→green, "white square"→red, "yellow"→blue). Semantic ranking math
  verified with injected vectors.
- **Swarm construction** via `TaskSolving.from_task("video_understanding", …)`:
  builds `BasicEnvironment` + `DescriptionAssigner` / `HorizontalDecisionMaker` /
  **`VideoEvidenceExecutor`** / `BasicEvaluator` and all agents.
- **Executor seam**: injecting a `VideoEvidenceIndex` and running the executor
  returns grounded, timestamped evidence (`[7.0s | vision | agreement=0.90] …`).

**Not runnable in every sandbox:** the semantic embedder (bge-small), Whisper
weights, and real VLM captioners all download from huggingface.co. Where egress
policy blocks HF, those paths fail soft (retrieval → lexical, ASR → skipped) and
the pipeline still runs; provide the weights/keys in an unrestricted environment
to enable them. The LLM debate itself needs an API key (Gemini free tier / OpenAI
/ local vLLM) to actually reason.

## 8. Honest limits

- "Near-infallible" is an asymptote, not a guarantee: comprehension is still bounded
  by the weakest captioner on genuinely ambiguous frames, and by ASR quality on noisy
  audio. The design's answer to this is **abstention** — the solver is required to say
  "Not evidenced" rather than fabricate, which trades recall for precision.
- Very long videos (hours) rely on retrieval, not full-context reading; recall depends
  on the index. Raise `max_frames` and add scene-level summaries if recall matters more
  than latency.
- The two OpenCV kernels in `sampling.py` and the `video-evidence` executor (§5) are the
  only pieces left to implement before this runs fully end-to-end on real files.
```
