"""Speech input. Optional: the app degrades to typing if this cannot load.

faster-whisper is used when present because it runs the small models on CPU at
a speed that is tolerable in a live demo. Hindi and English are both handled by
the multilingual checkpoints; we let the model auto-detect rather than forcing
a language, because users code-switch mid-sentence.
"""
from __future__ import annotations

import functools

from .config import ASR_MODEL


class ASRUnavailable(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def _model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRUnavailable(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    return WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_path: str) -> tuple[str, str]:
    """Return (text, detected_language)."""
    segments, info = _model().transcribe(audio_path, beam_size=1, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, getattr(info, "language", "")


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False
