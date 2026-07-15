"""Adaptive high-performance frame sampling.

Goal: maximum *effective* frame density (so nothing important is missed)
at minimum cost (so we do not caption thousands of near-identical frames).

Strategy — three complementary signals, cheap first:

1. Scene-cut detection (PySceneDetect / content detector): guarantees at
   least one keyframe per shot. This is what stops us from "losing" a frame
   where the content changes.
2. Motion / frame-difference sampling: inside a long static shot we still
   sample on motion spikes (a hand moves, text appears) instead of a fixed
   stride. This raises density exactly where information density is high.
3. Embedding de-duplication (CLIP / ImageBind): drop frames whose visual
   embedding is within `dedup_threshold` cosine of an already-kept frame.
   This removes redundancy the first two steps let through, keeping the
   caption budget for genuinely new content.

Every kept frame carries its exact timestamp so captions can be grounded in
time (critical for temporal reasoning downstream).

All heavy imports are lazy so importing this module never forces opencv /
scenedetect / torch to be installed until a pipeline actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from agentverse.logging import logger


@dataclass
class SampledFrame:
    """A single kept frame and everything needed to ground it in time."""

    index: int  # frame number in the source video
    timestamp: float  # seconds from start
    image_path: str  # extracted still on disk (jpg/png)
    scene_id: int  # which shot it belongs to
    reason: str = ""  # "scene-cut" | "motion" | "stride" (for auditability)
    embedding: Optional[list] = field(default=None, repr=False)


@dataclass
class SamplingConfig:
    # Floor / ceiling so density adapts but stays bounded for performance.
    min_fps: float = 0.2  # never sample slower than 1 frame / 5s
    max_fps: float = 4.0  # never sample faster than 4 frames / s
    scene_threshold: float = 27.0  # PySceneDetect content threshold
    motion_percentile: float = 75.0  # keep frames above this motion pctl in a shot
    dedup_threshold: float = 0.985  # cosine >= this => treated as duplicate
    max_frames: int = 2000  # hard cap for the whole video (budget guard)
    motion_floor: float = 1.0  # ignore sub-noise motion below this mean-abs-diff
    # "lite"  -> offline cv2 signature (grayscale + colour histogram), no torch
    # "clip"  -> open_clip embeddings if installed, else falls back to "lite"
    # "none"  -> skip dedup
    embedder: str = "lite"


class AdaptiveFrameSampler:
    """Turn a video file into a compact, timestamped set of keyframes.

    This is deliberately backend-pluggable and fails soft: if an optional
    dependency is missing it degrades to a coarser but still-correct method
    (e.g. uniform stride) and logs what it fell back to, so a partial install
    still produces a usable — just less dense — sampling.
    """

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()

    # -- public API -------------------------------------------------------
    def sample(self, video_path: str, out_dir: str) -> List[SampledFrame]:
        scenes = self._detect_scenes(video_path)
        frames = self._extract_keyframes(video_path, scenes, out_dir)
        frames = self._dedup(frames)
        if len(frames) > self.config.max_frames:
            logger.warn(
                f"[sampler] {len(frames)} frames exceeds max_frames="
                f"{self.config.max_frames}; keeping the most spread-out subset."
            )
            frames = self._budget_downselect(frames, self.config.max_frames)
        logger.info(f"[sampler] kept {len(frames)} frames from {video_path}")
        return frames

    # -- steps (stubs with real algorithm described) ----------------------
    def _detect_scenes(self, video_path: str):
        """Return list of (start_sec, end_sec) shots via PySceneDetect.

        Falls back to a single whole-video "scene" if scenedetect is absent.
        """
        try:
            from scenedetect import detect, ContentDetector  # noqa: F401

            scene_list = detect(
                video_path, ContentDetector(threshold=self.config.scene_threshold)
            )
            return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list] or [
                (0.0, self._duration(video_path))
            ]
        except Exception as e:  # pragma: no cover - optional dep / IO
            logger.warn(f"[sampler] scene detection unavailable ({e}); one-shot mode.")
            return [(0.0, self._duration(video_path))]

    def _extract_keyframes(self, video_path, scenes, out_dir):
        """Within each shot, sample on motion spikes between min/max fps.

        Single streaming decode pass (memory-safe on long video):
          * the first frame of every shot is always kept (reason="scene-cut");
          * a frame is force-kept once `max_gap = 1/min_fps` has elapsed since the
            last kept frame (reason="stride") so we never leave a long blind spot;
          * between `min_gap = 1/max_fps` and `max_gap`, a frame is kept when its
            motion (mean abs-diff vs the previous frame, on a downscaled gray copy)
            exceeds the running `motion_percentile` of the shot (reason="motion").
        This raises density exactly where the picture is changing and stays sparse
        where it is static — dense but cheap.
        """
        import os

        import cv2
        import numpy as np

        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        min_gap = 1.0 / max(self.config.max_fps, 1e-6)
        max_gap = 1.0 / max(self.config.min_fps, 1e-6)
        starts = [s for s, _ in scenes]

        def scene_of(t: float) -> int:
            import bisect

            i = bisect.bisect_right(starts, t) - 1
            return max(0, min(i, len(scenes) - 1))

        frames: List[SampledFrame] = []
        prev_gray: dict = {}
        last_kept: dict = {}
        motion_hist: dict = {}
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / fps
            sid = scene_of(t)
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            pg = prev_gray.get(sid)
            motion = 0.0 if pg is None else float(np.mean(cv2.absdiff(gray, pg)))
            prev_gray[sid] = gray
            mh = motion_hist.setdefault(sid, [])
            mh.append(motion)

            keep, reason = False, ""
            if sid not in last_kept:
                keep, reason = True, "scene-cut"
            else:
                gap = t - last_kept[sid]
                if gap >= max_gap:
                    keep, reason = True, "stride"
                elif gap >= min_gap:
                    thr = (
                        float(np.percentile(mh, self.config.motion_percentile))
                        if len(mh) >= 5
                        else 0.0
                    )
                    if motion >= thr and motion > self.config.motion_floor:
                        keep, reason = True, "motion"
            if keep:
                path = os.path.join(out_dir, f"f{idx:06d}_s{sid}.jpg")
                cv2.imwrite(path, frame)
                frames.append(
                    SampledFrame(
                        index=idx,
                        timestamp=round(t, 3),
                        image_path=path,
                        scene_id=sid,
                        reason=reason,
                    )
                )
                last_kept[sid] = t
        cap.release()
        return frames

    def _dedup(self, frames: List[SampledFrame]) -> List[SampledFrame]:
        """Drop near-duplicate frames by embedding cosine similarity.

        Greedy: keep a frame only if its max cosine to already-kept frames is
        below `dedup_threshold`. The embedding comes from `_embed_image`, which
        uses CLIP when available and otherwise a cheap offline cv2 signature.
        """
        if self.config.embedder == "none" or len(frames) < 2:
            return frames
        import numpy as np

        kept: List[SampledFrame] = []
        kept_vecs: List = []
        for f in frames:
            vec = self._embed_image(f.image_path)
            if vec is None:
                kept.append(f)
                continue
            if kept_vecs:
                sims = np.array([float(np.dot(vec, kv)) for kv in kept_vecs])
                if sims.max() >= self.config.dedup_threshold:
                    continue  # near-duplicate of a frame we already kept
            f.embedding = vec.tolist()
            kept.append(f)
            kept_vecs.append(vec)
        return kept

    def _embed_image(self, image_path: str):
        """Return an L2-normalised visual embedding for `image_path`.

        Prefers CLIP (`embedder="clip"` and open_clip installed); otherwise a
        fully-offline signature: 32x32 grayscale layout + per-channel colour
        histogram. Good enough to catch near-duplicate frames without any GPU.
        """
        import numpy as np

        if self.config.embedder == "clip":
            vec = self._clip_embed(image_path)
            if vec is not None:
                return vec  # else fall through to the lite signature
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return None

        def _unit(v):
            v = v.astype("float32")
            n = np.linalg.norm(v)
            return v / n if n else v

        # Spatial layout (standardised so absolute brightness doesn't dominate)
        # is the signal that reflects a moving object; colour histogram is a
        # secondary cue. Normalise EACH component before weighting, otherwise the
        # large-magnitude histogram counts wash the layout out entirely.
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 32)).astype(
            "float32"
        ).flatten()
        gray = gray - gray.mean()
        layout = _unit(gray)
        hist = _unit(
            np.concatenate(
                [cv2.calcHist([img], [c], None, [8], [0, 256]).flatten() for c in range(3)]
            )
        )
        vec = np.concatenate([0.75 * layout, 0.25 * hist]).astype("float32")
        return _unit(vec)

    def _clip_embed(self, image_path: str):
        try:
            import numpy as np
            import open_clip
            import torch
            from PIL import Image

            if not hasattr(self, "_clip"):
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                model.eval()
                self._clip = (model, preprocess)
            model, preprocess = self._clip
            with torch.no_grad():
                x = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
                v = model.encode_image(x)[0]
                v = v / v.norm()
                return v.cpu().numpy().astype("float32")
        except Exception:
            return None

    def _budget_downselect(self, frames, k):
        """Evenly thin frames over time to respect the hard budget cap."""
        if len(frames) <= k:
            return frames
        step = len(frames) / k
        return [frames[int(i * step)] for i in range(k)]

    def _duration(self, video_path) -> float:
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            return float(n) / float(fps) if fps else 0.0
        except Exception:
            return 0.0
