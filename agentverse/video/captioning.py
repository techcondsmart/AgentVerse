"""Dense, grounded frame captioning with an anti-hallucination ensemble.

This is where "loss of frames" is really fought. A single VLM pass on a frame
is both lossy (misses detail) and prone to hallucination (invents detail).
We counter both with a *caption ensemble*:

  1. Multi-model: caption each keyframe with >=2 independent VLM backends
     (e.g. Gemini Flash free-tier + a local Qwen-VL). Independent errors do
     not correlate, so agreement is strong evidence and disagreement flags a
     frame for the verifier swarm.
  2. Self-consistency: sample the same model a few times at low temperature
     and keep only detail that recurs across samples (majority grounding).
  3. Structured extraction: instead of one free-text blob, ask for typed
     fields — objects, actions, on-screen text (OCR), setting, and an explicit
     "uncertain" list. Typed output is far easier to cross-check and to feed
     retrieval than prose.
  4. Confidence + provenance: every caption keeps which model produced it and
     an agreement score, so downstream agents can weight evidence and never
     treat a low-agreement caption as fact.

The result per frame is a `FrameEvidence` record — the atomic, timestamped,
provenance-tagged unit the retrieval index and the reasoning swarm consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from agentverse.logging import logger
from agentverse.video.sampling import SampledFrame


@dataclass
class FrameEvidence:
    timestamp: float
    scene_id: int
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    ocr_text: str = ""
    setting: str = ""
    caption: str = ""  # fused natural-language summary of the fields
    uncertain: List[str] = field(default_factory=list)  # model-flagged low confidence
    agreement: float = 1.0  # 0..1 cross-model / self-consistency agreement
    sources: List[str] = field(default_factory=list)  # model ids that contributed


# A caption backend is any callable: (image_path, prompt) -> raw text.
CaptionBackend = Callable[[str, str], str]

STRUCTURED_PROMPT = (
    "Describe ONLY what is visibly present in this frame. Do not guess or infer "
    "beyond the pixels. Return JSON with keys: objects (list), actions (list), "
    "ocr_text (string, verbatim on-screen text or ''), setting (string), "
    "uncertain (list of anything you are not sure about). If nothing for a key, "
    "use an empty value. Never invent detail to fill a field."
)


class CaptionEnsemble:
    """Caption a set of frames using >=1 backend, with agreement scoring."""

    def __init__(
        self,
        backends: Dict[str, CaptionBackend],
        self_consistency_samples: int = 1,
    ):
        if not backends:
            raise ValueError("CaptionEnsemble needs at least one VLM backend.")
        self.backends = backends
        self.self_consistency_samples = max(1, self_consistency_samples)

    def caption_frames(self, frames: List[SampledFrame]) -> List[FrameEvidence]:
        out: List[FrameEvidence] = []
        for f in frames:
            per_model = {}
            for name, backend in self.backends.items():
                try:
                    per_model[name] = self._structured_call(backend, f.image_path)
                except Exception as e:  # pragma: no cover
                    logger.warn(f"[caption] backend {name} failed on {f.image_path}: {e}")
            out.append(self._fuse(f, per_model))
        return out

    def _structured_call(self, backend: CaptionBackend, image_path: str) -> dict:
        """Call one backend `self_consistency_samples` times and keep recurring detail."""
        import json

        samples = []
        for _ in range(self.self_consistency_samples):
            raw = backend(image_path, STRUCTURED_PROMPT)
            try:
                samples.append(json.loads(_strip_fences(raw)))
            except Exception:
                samples.append({"setting": raw.strip()})
        return _majority_merge(samples)

    def _fuse(self, frame: SampledFrame, per_model: Dict[str, dict]) -> FrameEvidence:
        """Fuse multiple models into one evidence record + agreement score."""
        if not per_model:
            return FrameEvidence(
                timestamp=frame.timestamp, scene_id=frame.scene_id, agreement=0.0
            )
        sources = list(per_model.keys())
        # Objects/actions kept only if they appear in a MAJORITY of models
        # (single-model-only detail is retained but marked uncertain).
        objects, obj_uncertain = _cross_model_consensus(per_model, "objects")
        actions, act_uncertain = _cross_model_consensus(per_model, "actions")
        agreement = _agreement_score(per_model)
        ocr = _pick_ocr(per_model)
        setting = _longest(per_model, "setting")
        ev = FrameEvidence(
            timestamp=frame.timestamp,
            scene_id=frame.scene_id,
            objects=objects,
            actions=actions,
            ocr_text=ocr,
            setting=setting,
            uncertain=obj_uncertain + act_uncertain,
            agreement=agreement,
            sources=sources,
        )
        ev.caption = _render_caption(ev)
        return ev


# -- small pure helpers (kept simple + testable) --------------------------
def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    return t.strip()


def _majority_merge(samples: List[dict]) -> dict:
    """Keep list items / strings that appear in > half the self-consistency samples."""
    if len(samples) == 1:
        return samples[0]
    merged: dict = {}
    keys = set().union(*[s.keys() for s in samples])
    for k in keys:
        vals = [s.get(k) for s in samples if s.get(k)]
        if vals and isinstance(vals[0], list):
            from collections import Counter

            c = Counter(x for v in vals for x in v)
            merged[k] = [x for x, n in c.items() if n > len(samples) / 2]
        else:
            merged[k] = max(vals, key=len) if vals else ""
    return merged


def _cross_model_consensus(per_model, key):
    from collections import Counter

    c = Counter()
    for m in per_model.values():
        for item in m.get(key, []) or []:
            c[str(item).lower()] += 1
    n = len(per_model)
    consensus = [x for x, k in c.items() if k > n / 2]
    uncertain = [x for x, k in c.items() if k <= n / 2]
    return consensus, uncertain


def _agreement_score(per_model) -> float:
    """Jaccard of object sets across models; 1.0 if a single model."""
    sets = [set((m.get("objects") or [])) for m in per_model.values()]
    sets = [{str(x).lower() for x in s} for s in sets if s]
    if len(sets) < 2:
        return 1.0
    inter = set.intersection(*sets)
    union = set.union(*sets)
    return len(inter) / len(union) if union else 1.0


def _pick_ocr(per_model):
    cands = [m.get("ocr_text", "") for m in per_model.values() if m.get("ocr_text")]
    return max(cands, key=len) if cands else ""


def _longest(per_model, key):
    cands = [m.get(key, "") for m in per_model.values() if m.get(key)]
    return max(cands, key=len) if cands else ""


def _render_caption(ev: FrameEvidence) -> str:
    parts = []
    if ev.setting:
        parts.append(ev.setting)
    if ev.objects:
        parts.append("Objects: " + ", ".join(ev.objects))
    if ev.actions:
        parts.append("Actions: " + ", ".join(ev.actions))
    if ev.ocr_text:
        parts.append(f'On-screen text: "{ev.ocr_text}"')
    return " | ".join(parts)
