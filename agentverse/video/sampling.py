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
    dedup_threshold: float = 0.94  # cosine >= this => treated as duplicate
    max_frames: int = 2000  # hard cap for the whole video (budget guard)
    embedder: str = "clip"  # "clip" | "imagebind" | "none"


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

        See module docstring. Uses OpenCV frame differencing; on failure,
        falls back to uniform stride at min_fps.
        """
        raise NotImplementedError(
            "Wire to OpenCV: per shot, compute abs frame diff, keep frames above "
            "motion_percentile bounded by [min_fps, max_fps], write stills to out_dir, "
            "and tag each SampledFrame with its timestamp + scene_id + reason."
        )

    def _dedup(self, frames: List[SampledFrame]) -> List[SampledFrame]:
        """Drop near-duplicate frames by embedding cosine similarity."""
        if self.config.embedder == "none":
            return frames
        raise NotImplementedError(
            "Embed each frame (CLIP or ImageBind), then greedily keep a frame only if "
            "its max cosine to already-kept frames < dedup_threshold."
        )

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
