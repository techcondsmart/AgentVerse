"""Video comprehension subsystem for AgentVerse.

Perception (this package) turns a raw video into a frozen, timestamped,
provenance-tagged evidence index; the reasoning swarm
(agentverse/tasks/tasksolving/video_understanding) then debates and verifies
answers over that fixed evidence.

Typical usage:

    from agentverse.video import VideoPerceptionPipeline, SamplingConfig
    from agentverse.video.backends import gemini_backend, local_vllm_backend

    pipeline = VideoPerceptionPipeline(
        caption_backends={
            "gemini": gemini_backend(),                       # free-tier, native video
            "qwen-vl": local_vllm_backend("Qwen/Qwen2.5-VL-7B-Instruct"),  # open-weights 2nd opinion
        },
        sampling=SamplingConfig(max_fps=4.0, dedup_threshold=0.94),
        self_consistency_samples=3,   # keep only detail that recurs across samples
    )
    knowledge = pipeline.process("lecture.mp4", work_dir="./_work")

    # Hand knowledge.index to the swarm (see documentation/video_understanding.md).
    hits = knowledge.index.query("What did the speaker say about safety?")
"""

from agentverse.video.asr import Transcriber, TranscriptSegment
from agentverse.video.captioning import CaptionEnsemble, FrameEvidence
from agentverse.video.pipeline import VideoKnowledge, VideoPerceptionPipeline
from agentverse.video.retrieval import EvidenceHit, VideoEvidenceIndex
from agentverse.video.sampling import (
    AdaptiveFrameSampler,
    SampledFrame,
    SamplingConfig,
)

__all__ = [
    "VideoPerceptionPipeline",
    "VideoKnowledge",
    "AdaptiveFrameSampler",
    "SamplingConfig",
    "SampledFrame",
    "Transcriber",
    "TranscriptSegment",
    "CaptionEnsemble",
    "FrameEvidence",
    "VideoEvidenceIndex",
    "EvidenceHit",
]
