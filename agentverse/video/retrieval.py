"""VideoRAG-style grounded retrieval over frame evidence + transcript.

Instead of stuffing every caption into the LLM context (lossy, expensive, and
a hallucination magnet), we build a timestamped evidence index and retrieve
only the segments relevant to a query. This is the mechanism that lets the
swarm answer with CITATIONS ("at 03:12 the sign reads ...") rather than vibes.

Two indexes, fused at query time:
  - Semantic: embed each FrameEvidence.caption and each TranscriptSegment.text
    into a vector store (reuses AgentVerse's existing vectorstore memory).
  - Temporal: keep everything sorted by timestamp so we can expand a hit into
    its neighbourhood and reason about ORDER and DURATION (temporal reasoning).

A retrieval result always carries timestamps and provenance so the reasoning
agents can quote and the verifier agents can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agentverse.video.asr import TranscriptSegment
from agentverse.video.captioning import FrameEvidence


@dataclass
class EvidenceHit:
    timestamp: float
    modality: str  # "vision" | "audio"
    text: str
    score: float
    agreement: float = 1.0  # visual agreement / 1.0 for transcript
    source: str = ""


class VideoEvidenceIndex:
    """Fused semantic + temporal index over one video's evidence."""

    def __init__(self, embedder: str = "bge-small"):
        self.embedder = embedder
        self._items: List[EvidenceHit] = []
        self._vectors = None  # lazily built matrix

    def build(
        self,
        frames: List[FrameEvidence],
        transcript: List[TranscriptSegment],
    ) -> "VideoEvidenceIndex":
        self._items = []
        for f in frames:
            if f.caption:
                self._items.append(
                    EvidenceHit(f.timestamp, "vision", f.caption, 0.0,
                                agreement=f.agreement,
                                source=",".join(f.sources))
                )
        for s in transcript:
            self._items.append(
                EvidenceHit(s.start, "audio", s.text, 0.0, source="asr")
            )
        self._items.sort(key=lambda h: h.timestamp)
        self._embed_corpus()
        return self

    def query(
        self, question: str, k: int = 8, window: float = 4.0
    ) -> List[EvidenceHit]:
        """Return top-k semantically relevant hits, each expanded temporally.

        The temporal window pulls in neighbouring evidence so the reasoner sees
        what happened just before/after — essential for order & causality.
        """
        ranked = self._semantic_rank(question, k)
        return self._expand_temporal(ranked, window)

    def timeline(self) -> List[EvidenceHit]:
        """Full ordered evidence — used by the summarizer / temporal agent."""
        return list(self._items)

    # -- backend hooks (lazy) --------------------------------------------
    def _embed_corpus(self):
        """Embed all item texts with a free local embedder (sentence-transformers).

        Kept optional: if unavailable, query() falls back to lexical BM25-ish
        scoring so retrieval still works, just less semantically.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self._model = model
            self._vectors = model.encode(
                [it.text for it in self._items], normalize_embeddings=True
            )
        except Exception:
            self._model = None
            self._vectors = None

    def _semantic_rank(self, question, k):
        if self._vectors is not None:
            import numpy as np

            q = self._model.encode([question], normalize_embeddings=True)[0]
            sims = self._vectors @ q
            order = sims.argsort()[::-1][:k]
            hits = []
            for i in order:
                h = self._items[int(i)]
                hits.append(EvidenceHit(h.timestamp, h.modality, h.text,
                                        float(sims[int(i)]), h.agreement, h.source))
            return hits
        return self._lexical_rank(question, k)

    def _lexical_rank(self, question, k):
        import re

        def toks(s):
            return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}

        terms = toks(question)
        scored = []
        for it in self._items:
            overlap = len(terms & toks(it.text))
            if overlap:
                scored.append(EvidenceHit(it.timestamp, it.modality, it.text,
                                          float(overlap), it.agreement, it.source))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def _expand_temporal(self, hits, window):
        keep, seen = [], set()
        for h in hits:
            for it in self._items:
                if abs(it.timestamp - h.timestamp) <= window:
                    key = (round(it.timestamp, 2), it.modality)
                    if key not in seen:
                        seen.add(key)
                        keep.append(it)
        keep.sort(key=lambda x: x.timestamp)
        return keep
