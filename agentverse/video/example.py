"""Minimal end-to-end example of the video comprehension subsystem.

Run:  python -m agentverse.video.example path/to/video.mp4 "your question"

This wires perception -> retrieval and prints grounded, timestamped evidence for
a question. The reasoning swarm (agentverse/tasks/tasksolving/video_understanding)
consumes exactly this evidence; see documentation/video_understanding.md §5 for
binding it to the AgentVerse executor.

Requires the extras in requirements-video.txt and, for hosted captioning, a
GEMINI_API_KEY (or swap in a local backend). The two OpenCV kernels in
sampling.py are stubs — this example documents the intended flow and will run
fully once they are filled in.
"""

from __future__ import annotations

import os
import sys

from agentverse.video import SamplingConfig, VideoPerceptionPipeline
from agentverse.video.backends import gemini_backend, local_vllm_backend


def build_pipeline() -> VideoPerceptionPipeline:
    # Pair an independent hosted + local captioner so their errors don't correlate.
    backends = {}
    if os.environ.get("GEMINI_API_KEY"):
        backends["gemini"] = gemini_backend()  # free-tier, native video captioner
    if os.environ.get("VLLM_BASE_URL"):
        backends["qwen-vl"] = local_vllm_backend(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            base_url=os.environ["VLLM_BASE_URL"],
        )
    if not backends:
        raise SystemExit(
            "Set GEMINI_API_KEY and/or VLLM_BASE_URL to enable a caption backend."
        )
    return VideoPerceptionPipeline(
        caption_backends=backends,
        sampling=SamplingConfig(max_fps=4.0, dedup_threshold=0.94, max_frames=1500),
        self_consistency_samples=3,  # keep only detail that recurs across samples
    )


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    video_path, question = sys.argv[1], sys.argv[2]

    pipeline = build_pipeline()
    knowledge = pipeline.process(video_path, work_dir="./_video_work")

    print(f"\nPerceived {knowledge.n_frames} frames, "
          f"{knowledge.n_transcript_segments} speech segments. "
          f"{len(knowledge.low_agreement_timestamps)} low-agreement frames flagged.\n")

    print(f"Q: {question}\nGrounded evidence:")
    for hit in knowledge.index.query(question, k=8):
        print(f"  [{hit.timestamp:6.1f}s | {hit.modality:6s} | "
              f"agreement={hit.agreement:.2f}] {hit.text}")


if __name__ == "__main__":
    main()
