"""Executor that grounds the swarm in a video's evidence index.

The critics of the video_understanding swarm demand evidence ("is object X
actually visible at 00:42?"). This executor runs their queries against the
frozen `VideoEvidenceIndex` built by agentverse/video/pipeline.py and returns
the matching timestamped, provenance-tagged hits — the retrieval half of the
VideoRAG loop.

Wiring: build a VideoKnowledge with the perception pipeline, then inject its
index before running the task, e.g.

    from agentverse.environments.tasksolving_env.rules.executor.video_evidence \
        import VideoEvidenceExecutor
    VideoEvidenceExecutor.index = knowledge.index

or set it on the constructed executor instance. In the config use
`executor: {type: video-evidence}`.

If no index is bound it fails soft (returns an empty evidence message) so the
swarm still runs on evidence passed inline via the prompt.
"""

from __future__ import annotations

from typing import Any, List, Optional

from agentverse.agents import ExecutorAgent
from agentverse.message import ExecutorMessage, SolverMessage

from . import executor_registry
from .base import BaseExecutor


@executor_registry.register("video-evidence")
class VideoEvidenceExecutor(BaseExecutor):
    """Retrieve timestamped evidence for the queries in the current solution."""

    # Injected before run: a agentverse.video.retrieval.VideoEvidenceIndex.
    # Class-level so it can be set once for the whole task; also settable
    # per-instance. Typed as Any to avoid importing torch/cv2 at module load.
    index: Optional[Any] = None
    top_k: int = 8
    window: float = 4.0

    def step(
        self,
        agent: ExecutorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return self._retrieve(solution)

    async def astep(
        self,
        agent: ExecutorAgent,
        task_description: str,
        solution: List[SolverMessage],
        *args,
        **kwargs,
    ) -> Any:
        return self._retrieve(solution)

    # -- core -------------------------------------------------------------
    def _retrieve(self, solution) -> List[ExecutorMessage]:
        if self.index is None:
            return [
                ExecutorMessage(
                    content="[no evidence index bound — reasoning on inline evidence]"
                )
            ]
        queries = self._queries_from(solution)
        blocks = []
        for q in queries:
            hits = self.index.query(q, k=self.top_k, window=self.window)
            lines = [self._fmt(h) for h in hits] or ["(no matching evidence)"]
            blocks.append(f"### Evidence for: {q}\n" + "\n".join(lines))
        return [ExecutorMessage(content="\n\n".join(blocks))]

    @staticmethod
    def _queries_from(solution) -> List[str]:
        """Each non-empty line of the solution is treated as a retrieval query."""
        text = "\n".join(
            getattr(s, "content", str(s)) for s in (solution or [])
        )
        qs = [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()]
        return qs or [text.strip()] if text.strip() else []

    @staticmethod
    def _fmt(hit) -> str:
        return (
            f"[{hit.timestamp:0.1f}s | {hit.modality} | "
            f"agreement={hit.agreement:.2f}] {hit.text}"
        )
