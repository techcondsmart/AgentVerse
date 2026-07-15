"""Speech-to-text with word/segment timestamps.

Audio is a first-class evidence channel: much of a video's meaning is spoken,
not visible. We transcribe with Whisper-family models and keep timestamps so
every claim the swarm makes can be grounded to "who said what, when".

Backends, in order of preference (all free / open-weights):
  - faster-whisper (CTranslate2)  -> fastest local, low VRAM, word timestamps
  - openai-whisper                -> reference implementation, CPU-capable
  - Groq Whisper API (free tier)  -> no local GPU needed; see llms/multimodal

Fails soft: if no backend is available the pipeline still runs on the visual
channel alone, but logs that audio evidence is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from agentverse.logging import logger


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None  # filled if diarization is enabled


class Transcriber:
    def __init__(self, model_size: str = "large-v3", backend: str = "faster-whisper"):
        self.model_size = model_size
        self.backend = backend

    def transcribe(self, audio_or_video_path: str) -> List[TranscriptSegment]:
        try:
            if self.backend == "faster-whisper":
                return self._faster_whisper(audio_or_video_path)
            return self._openai_whisper(audio_or_video_path)
        except Exception as e:  # pragma: no cover - optional dep
            logger.warn(
                f"[asr] transcription unavailable ({e}); continuing on visual channel only."
            )
            return []

    def _faster_whisper(self, path) -> List[TranscriptSegment]:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_size, compute_type="auto")
        segments, _ = model.transcribe(path, word_timestamps=True, vad_filter=True)
        return [TranscriptSegment(s.start, s.end, s.text.strip()) for s in segments]

    def _openai_whisper(self, path) -> List[TranscriptSegment]:
        import whisper

        model = whisper.load_model(self.model_size)
        result = model.transcribe(path)
        return [
            TranscriptSegment(s["start"], s["end"], s["text"].strip())
            for s in result.get("segments", [])
        ]
