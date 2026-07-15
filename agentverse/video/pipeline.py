"""End-to-end preprocessing: raw video -> queryable evidence index.

This is the "perception" stage that runs ONCE per video, before the reasoning
swarm ever sees it. Keeping perception separate from reasoning is deliberate:
the expensive, non-deterministic vision/audio work is done, grounded, and
frozen into an auditable evidence store; the multi-agent debate then reasons
over that fixed evidence, so its conclusions are reproducible and checkable.

    video.mp4
       |-- AdaptiveFrameSampler  -> timestamped keyframes (no lost content)
       |-- Transcriber (Whisper)  -> timestamped speech
       |-- CaptionEnsemble        -> grounded, provenance-tagged FrameEvidence
       '-- VideoEvidenceIndex     -> fused semantic + temporal store
                                     (this object is handed to the swarm)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from agentverse.logging import logger
from agentverse.video.asr import Transcriber
from agentverse.video.captioning import CaptionBackend, CaptionEnsemble
from agentverse.video.retrieval import VideoEvidenceIndex
from agentverse.video.sampling import AdaptiveFrameSampler, SamplingConfig


@dataclass
class VideoKnowledge:
    """Everything the reasoning swarm needs, frozen and auditable."""

    video_path: str
    index: VideoEvidenceIndex
    n_frames: int
    n_transcript_segments: int
    low_agreement_timestamps: list  # frames the verifier should scrutinize


class VideoPerceptionPipeline:
    def __init__(
        self,
        caption_backends: Dict[str, CaptionBackend],
        sampling: Optional[SamplingConfig] = None,
        whisper_size: str = "large-v3",
        self_consistency_samples: int = 1,
        low_agreement_threshold: float = 0.5,
    ):
        self.sampler = AdaptiveFrameSampler(sampling)
        self.transcriber = Transcriber(model_size=whisper_size)
        self.ensemble = CaptionEnsemble(caption_backends, self_consistency_samples)
        self.low_agreement_threshold = low_agreement_threshold

    def process(self, video_path: str, work_dir: str) -> VideoKnowledge:
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"[pipeline] perceiving {video_path}")

        frames = self.sampler.sample(video_path, os.path.join(work_dir, "frames"))
        transcript = self.transcriber.transcribe(video_path)
        evidence = self.ensemble.caption_frames(frames)

        low_conf = [
            e.timestamp for e in evidence if e.agreement < self.low_agreement_threshold
        ]
        index = VideoEvidenceIndex().build(evidence, transcript)

        logger.info(
            f"[pipeline] done: {len(evidence)} frames, {len(transcript)} speech "
            f"segments, {len(low_conf)} low-agreement frames flagged for verification."
        )
        return VideoKnowledge(
            video_path=video_path,
            index=index,
            n_frames=len(evidence),
            n_transcript_segments=len(transcript),
            low_agreement_timestamps=low_conf,
        )
