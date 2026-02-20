import re
import os

import numpy as np

# ── Compatibility shim ─────────────────────────────────────────────────────
# kokoro imports `is_offline_mode` from huggingface_hub, which was removed in
# huggingface_hub >= 0.27. Inject it back before kokoro is loaded anywhere.
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, "is_offline_mode"):
    _hf_hub.is_offline_mode = lambda: os.getenv("HF_HUB_OFFLINE", "0") == "1"
# ──────────────────────────────────────────────────────────────────────────


# Cache the Kokoro pipeline so it's only loaded once
_kokoro_pipeline = None

_EMOJI_TTS_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # flags
    "\U0001F300-\U0001FAFF"  # emoji and pictographs
    "\U00002700-\U000027BF"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters and misc emoji
    "\u200d"                 # zero-width joiner
    "\uFE0E\uFE0F"          # variation selectors
    "]+",
    flags=re.UNICODE,
)


def _strip_emoji_for_tts(text: str) -> str:
    cleaned = _EMOJI_TTS_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def _get_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    return _kokoro_pipeline


def speak(text: str, voice: str = 'af_heart', speed: float = 1.0) -> None:
    """
    Convert text to speech using Kokoro TTS and play it through speakers.
    Runs fully locally — no internet required.

    Args:
        text: The text to speak aloud
        voice: Kokoro voice ID. American English options: 'af_heart', 'af_sky', 'af_bella'
        speed: Speech speed multiplier (1.0 = normal, 1.2 = slightly faster)
    """
    try:
        import sounddevice as sd

        text = _strip_emoji_for_tts(text)
        if not text:
            return

        pipeline = _get_pipeline()
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+')

        for i, (gs, ps, audio) in enumerate(generator):
            # audio is a numpy float32 array, sample rate is 24000 Hz
            sd.play(audio, samplerate=24000)
            sd.wait()  # block until audio finishes before playing next chunk

    except ImportError as ie:
        print(f"[TTS] Missing dependency: {ie}. Run: pip install kokoro sounddevice")
    except Exception as e:
        print(f"[TTS] Error: {e}")


async def speak_async(text: str, voice: str = 'af_heart', speed: float = 1.0) -> None:
    """
    Async wrapper for speak() — use this from FastAPI/async contexts.
    Runs speak() in a thread pool to avoid blocking the event loop.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, speak, text, voice, speed)


def speak_sentence(text: str, voice: str = 'af_heart', speed: float = 1.0) -> None:
    """
    Speak a single sentence/chunk immediately. Blocks until audio finishes.
    Designed to be called from a background thread while the LLM is still streaming.
    """
    try:
        import sounddevice as sd
        text = _strip_emoji_for_tts(text)
        if not text:
            return

        pipeline = _get_pipeline()
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+'):
            sd.play(audio, samplerate=24000)
            sd.wait()
    except Exception as e:
        print(f"[TTS] speak_sentence error: {e}")


def get_sentence_audio_bytes(text: str, voice: str = 'af_heart', speed: float = 1.0) -> bytes:
    """
    Generate TTS audio for a single sentence and return as WAV bytes.
    Designed for streaming sentence-by-sentence to the browser.
    """
    import io
    import wave

    try:
        text = _strip_emoji_for_tts(text)
        if not text:
            return b""

        pipeline = _get_pipeline()
        all_audio = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+'):
            all_audio.append(audio)

        if not all_audio:
            return b""

        combined = np.concatenate(all_audio)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(24000)
            wf.writeframes((combined * 32767).astype(np.int16).tobytes())
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS] get_sentence_audio_bytes error: {e}")
        return b""


def get_audio_bytes(text: str, voice: str = 'af_heart', speed: float = 1.0) -> bytes:
    """
    Generate TTS audio and return as raw bytes (WAV format).
    Use this if you need to stream audio to the frontend instead of playing locally.
    """
    import io
    import wave

    try:
        text = _strip_emoji_for_tts(text)
        if not text:
            return b""

        pipeline = _get_pipeline()
        all_audio = []

        for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=speed)):
            all_audio.append(audio)

        if not all_audio:
            return b""

        combined = np.concatenate(all_audio)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16-bit
            wf.setframerate(24000)
            wf.writeframes((combined * 32767).astype(np.int16).tobytes())
        return buf.getvalue()

    except Exception as e:
        print(f"[TTS] get_audio_bytes error: {e}")
        return b""
