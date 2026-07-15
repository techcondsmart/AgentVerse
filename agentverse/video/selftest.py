"""Offline self-test for the video comprehension subsystem.

Runs the whole perception + retrieval + executor flow on a synthetically
generated video with *mock* caption backends, so it needs no API keys and no
downloaded model weights — everything runs from cv2 + numpy alone.

    python -m agentverse.video.selftest

It validates:
  * adaptive frame sampling (scene-cut + motion + embedding dedup) on real pixels,
  * the multi-model caption ensemble + agreement scoring,
  * the fused semantic/temporal evidence index (lexical fallback path),
  * the `video-evidence` executor retrieving grounded, timestamped evidence.

The semantic-embedding path (sentence-transformers), Whisper ASR, and real VLM
captioners are exercised in production but require network/model access; this
self-test deliberately avoids them so it is hermetic and fast.
"""

from __future__ import annotations

import json
import os
import tempfile


def _make_video(path: str, fps: int = 10):
    import cv2
    import numpy as np

    W, H = 320, 240
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    def bg(color):
        f = np.zeros((H, W, 3), np.uint8)
        f[:] = color
        return f

    for i in range(40):  # scene A: red, moving white square (motion)
        f = bg((0, 0, 180))
        x = 10 + i * 7
        cv2.rectangle(f, (x, 90), (x + 40, 150), (255, 255, 255), -1)
        vw.write(f)
    for _ in range(30):  # scene B: blue, static
        f = bg((180, 0, 0))
        cv2.circle(f, (160, 120), 40, (0, 255, 255), -1)
        vw.write(f)
    for i in range(30):  # scene C: green, text + moving bar
        f = bg((0, 140, 0))
        cv2.putText(f, "SAFETY FIRST", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        y = 20 + i * 6
        cv2.rectangle(f, (0, y), (W, y + 8), (0, 0, 0), -1)
        vw.write(f)
    vw.release()


def _dominant(image_path: str) -> str:
    import cv2

    img = cv2.imread(image_path)
    b, g, r = [int(img[:, :, c].mean()) for c in range(3)]
    if r > 120 and g < 100:
        return "red"
    if b > 120 and r < 100:
        return "blue"
    if g > 100 and r < 100:
        return "green"
    return "mixed"


def _mock_backend(extra):
    """A deterministic offline stand-in for a real VLM caption backend."""
    table = {
        "red": {"setting": "red background", "objects": ["white square"],
                "actions": ["a square moves across the frame"], "ocr_text": ""},
        "blue": {"setting": "blue background", "objects": ["yellow circle"],
                 "actions": [], "ocr_text": ""},
        "green": {"setting": "green background", "objects": ["black bar"],
                  "actions": ["a bar slides down"], "ocr_text": "SAFETY FIRST"},
        "mixed": {"setting": "", "objects": [], "actions": [], "ocr_text": ""},
    }

    def backend(image_path: str, prompt: str) -> str:
        col = _dominant(image_path)
        data = dict(table[col])
        data["objects"] = data["objects"] + extra.get(col, [])
        data["uncertain"] = []
        return json.dumps(data)

    return backend


def main() -> int:
    from agentverse.video import SamplingConfig, VideoPerceptionPipeline

    work = tempfile.mkdtemp(prefix="videoselftest_")
    video = os.path.join(work, "test.mp4")
    _make_video(video)

    pipeline = VideoPerceptionPipeline(
        caption_backends={
            "mockA": _mock_backend({}),
            "mockB": _mock_backend({"red": ["floor"], "green": ["text overlay"]}),
        },
        sampling=SamplingConfig(min_fps=0.5, max_fps=6.0, motion_percentile=70,
                                dedup_threshold=0.985),
        self_consistency_samples=1,
    )
    knowledge = pipeline.process(video, work)
    assert knowledge.n_frames > 5, "sampler kept too few frames"

    # retrieval: each query must surface its own scene
    checks = {
        "safety warning text on screen": "SAFETY FIRST",
        "the white square that moves": "white square",
        "a round yellow shape": "yellow circle",
    }
    for query, needle in checks.items():
        hits = knowledge.index.query(query, k=3, window=0.0)
        joined = " ".join(h.text for h in hits)
        assert needle.lower() in joined.lower(), f"{query!r} did not surface {needle!r}"
        print(f"  ✓ query {query!r} -> {hits[0].timestamp:.1f}s: {hits[0].text[:50]}")

    # executor seam
    from agentverse.environments.tasksolving_env.rules.executor import (
        executor_registry,
    )
    from agentverse.message import SolverMessage

    ex = executor_registry.build("video-evidence")
    ex.index = knowledge.index
    out = ex.step(None, "q", [SolverMessage(content="safety warning text")])
    assert "SAFETY FIRST" in out[0].content, "executor failed to retrieve evidence"

    print(f"\nSELF-TEST PASSED: {knowledge.n_frames} frames, "
          f"retrieval + executor grounded correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
