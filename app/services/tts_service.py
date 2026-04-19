import asyncio
import io
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PIPER_MODEL = _PROJECT_ROOT / "models" / "piper" / "en_GB-cori-high.onnx"

_PIPER_VOICE = None
_PIPER_VOICE_KEY: Optional[tuple[str, str]] = None
_PIPER_VOICE_LOCK = threading.Lock()

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

_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_PATTERN = re.compile(r"https?://\S+")
_LIST_PREFIX_PATTERN = re.compile(r"(?m)^\s*[-*+]\s+")
_QUOTE_PREFIX_PATTERN = re.compile(r"(?m)^\s*>+\s*")
_HEADING_PREFIX_PATTERN = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_STANDALONE_DASH_PATTERN = re.compile(r"(?<!\w)-(?!\w)")
_MARKUP_SYMBOL_PATTERN = re.compile(r"[*`~_|]")


def _strip_emoji_for_tts(text: str) -> str:
    cleaned = _EMOJI_TTS_PATTERN.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_markup_symbols_for_tts(text: str) -> str:
    cleaned = _MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    cleaned = _URL_PATTERN.sub(" ", cleaned)
    cleaned = _LIST_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _QUOTE_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _HEADING_PREFIX_PATTERN.sub("", cleaned)
    cleaned = _STANDALONE_DASH_PATTERN.sub(" ", cleaned)
    cleaned = _MARKUP_SYMBOL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _sanitize_text_for_tts(text: str) -> str:
    # Drop invalid surrogate code points and normalize whitespace.
    safe_text = str(text or "")
    safe_text = safe_text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    safe_text = "".join(
        ch if (ch == "\n" or ch == "\t" or ord(ch) >= 32) else " " for ch in safe_text
    )
    safe_text = _strip_emoji_for_tts(safe_text)
    return _strip_markup_symbols_for_tts(safe_text)


def _parse_env_float(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None

    try:
        return float(raw)
    except ValueError:
        print(f"[TTS] Invalid {name}={raw!r}. Ignoring it.")
        return None


def _resolve_speed(speed: float) -> float:
    resolved = speed
    if speed == 1.0:
        env_speed = _parse_env_float("PIPER_SPEED")
        if env_speed is not None:
            resolved = env_speed

    if resolved <= 0:
        resolved = 1.0

    return max(0.5, min(resolved, 2.0))


def _resolve_sentence_silence() -> float:
    sentence_silence = _parse_env_float("PIPER_SENTENCE_SILENCE")
    if sentence_silence is None:
        return 0.0
    return max(0.0, min(sentence_silence, 1.0))


def _resolve_piper_binary() -> str:
    return os.getenv("PIPER_BIN", "piper").strip() or "piper"


def _resolve_piper_invocation() -> list[str]:
    configured = _resolve_piper_binary()
    configured_on_path = shutil.which(configured)
    if configured_on_path:
        return [configured_on_path]

    if configured != "piper":
        configured_path = Path(configured)
        if configured_path.exists():
            return [str(configured_path)]

    if importlib.util.find_spec("piper") is not None:
        return [sys.executable, "-m", "piper"]

    raise FileNotFoundError(
        "Piper runtime was not found. Install system 'piper' binary or install "
        "the Python package with 'pip install piper-tts'."
    )


def _import_piper_api():
    try:
        from piper.config import SynthesisConfig
        from piper.voice import PiperVoice

        return PiperVoice, SynthesisConfig
    except Exception as exc:
        raise ImportError(
            "Piper Python API was not found. Install it with 'pip install piper-tts'."
        ) from exc


def _resolve_model_path() -> Path:
    configured = os.getenv("PIPER_MODEL_PATH", "").strip()
    model_path = Path(configured).expanduser() if configured else _DEFAULT_PIPER_MODEL
    if not model_path.is_absolute():
        model_path = (_PROJECT_ROOT / model_path).resolve()
    return model_path


def _resolve_config_path(model_path: Path) -> Optional[Path]:
    configured = os.getenv("PIPER_CONFIG_PATH", "").strip()
    if configured:
        config_path = Path(configured).expanduser()
        if not config_path.is_absolute():
            config_path = (_PROJECT_ROOT / config_path).resolve()
        return config_path

    auto_config = model_path.with_suffix(model_path.suffix + ".json")
    return auto_config if auto_config.exists() else None


def _resolve_speaker_id(voice: str) -> Optional[int]:
    speaker_env = os.getenv("PIPER_SPEAKER_ID", "").strip()
    if speaker_env:
        try:
            return int(speaker_env)
        except ValueError:
            print(f"[TTS] Invalid PIPER_SPEAKER_ID={speaker_env!r}. Ignoring it.")

    if voice:
        match = re.search(r"\d+", voice)
        if match:
            return int(match.group(0))
    return None


def _speed_to_length_scale(speed: float) -> float:
    effective_speed = _resolve_speed(speed)

    # Piper uses length_scale where smaller values are faster.
    length_scale = 1.0 / effective_speed
    return max(0.5, min(length_scale, 2.0))


def _load_piper_voice():
    global _PIPER_VOICE
    global _PIPER_VOICE_KEY

    model_path = _resolve_model_path()
    config_path = _resolve_config_path(model_path)
    key = (str(model_path), str(config_path) if config_path else "")

    with _PIPER_VOICE_LOCK:
        if (_PIPER_VOICE is not None) and (_PIPER_VOICE_KEY == key):
            return _PIPER_VOICE

        PiperVoice, _ = _import_piper_api()
        _PIPER_VOICE = PiperVoice.load(
            model_path=str(model_path),
            config_path=str(config_path) if config_path else None,
            use_cuda=False,
        )
        _PIPER_VOICE_KEY = key
        return _PIPER_VOICE


def _build_synthesis_config(voice: str, speed: float):
    _, SynthesisConfig = _import_piper_api()
    return SynthesisConfig(
        speaker_id=_resolve_speaker_id(voice),
        length_scale=_speed_to_length_scale(speed),
        noise_scale=_parse_env_float("PIPER_NOISE_SCALE"),
        noise_w_scale=_parse_env_float("PIPER_NOISE_W"),
        normalize_audio=True,
        volume=1.0,
    )


def _ensure_piper_ready() -> None:
    # Prefer in-process Piper API if available; otherwise verify CLI/module fallback.
    if importlib.util.find_spec("piper") is None:
        _resolve_piper_invocation()

    model_path = _resolve_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            "Piper model not found. "
            f"Expected at: {model_path}. Set PIPER_MODEL_PATH to your .onnx model."
        )

    config_path = _resolve_config_path(model_path)
    if config_path and not config_path.exists():
        raise FileNotFoundError(
            f"Piper config not found at: {config_path}. "
            "Set PIPER_CONFIG_PATH correctly or remove it to auto-detect."
        )


def _build_piper_command(output_wav: Path, voice: str, speed: float) -> list[str]:
    model_path = _resolve_model_path()
    config_path = _resolve_config_path(model_path)
    speaker_id = _resolve_speaker_id(voice)

    cmd = [
        *_resolve_piper_invocation(),
        "--model",
        str(model_path),
        "--output_file",
        str(output_wav),
        "--length_scale",
        f"{_speed_to_length_scale(speed):.3f}",
    ]

    if config_path:
        cmd.extend(["--config", str(config_path)])

    if speaker_id is not None:
        cmd.extend(["--speaker", str(speaker_id)])

    noise_scale = os.getenv("PIPER_NOISE_SCALE", "").strip()
    if noise_scale:
        cmd.extend(["--noise_scale", noise_scale])

    noise_w = os.getenv("PIPER_NOISE_W", "").strip()
    if noise_w:
        cmd.extend(["--noise_w", noise_w])

    sentence_silence = _resolve_sentence_silence()
    if sentence_silence > 0:
        cmd.extend(["--sentence_silence", f"{sentence_silence:.3f}"])

    return cmd


def _synthesize_piper_wav_inprocess(text: str, voice: str, speed: float) -> bytes:
    text = _sanitize_text_for_tts(text)
    if not text:
        return b""

    piper_voice = _load_piper_voice()
    syn_config = _build_synthesis_config(voice, speed)
    sentence_silence = _resolve_sentence_silence()

    audio_chunks = list(piper_voice.synthesize(text, syn_config=syn_config))
    if not audio_chunks:
        return b""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        first_chunk = audio_chunks[0]
        wav_file.setframerate(first_chunk.sample_rate)
        wav_file.setsampwidth(first_chunk.sample_width)
        wav_file.setnchannels(first_chunk.sample_channels)

        for index, audio_chunk in enumerate(audio_chunks):
            wav_file.writeframes(audio_chunk.audio_int16_bytes)

            if sentence_silence > 0 and index < (len(audio_chunks) - 1):
                silence_samples = int(round(audio_chunk.sample_rate * sentence_silence))
                if silence_samples > 0:
                    silence_bytes = b"\x00" * (
                        silence_samples * audio_chunk.sample_width * audio_chunk.sample_channels
                    )
                    wav_file.writeframes(silence_bytes)

    return buffer.getvalue()


def _synthesize_piper_wav_subprocess(text: str, voice: str, speed: float) -> bytes:
    text = _sanitize_text_for_tts(text)
    if not text:
        return b""

    _ensure_piper_ready()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = Path(tmp.name)

    cmd = _build_piper_command(output_path, voice, speed)

    try:
        proc = subprocess.run(
            cmd,
            input=(text + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(
                f"Piper failed with exit code {proc.returncode}: {stderr or 'no stderr output'}"
            )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Piper did not generate a WAV file.")

        return output_path.read_bytes()
    finally:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass


def _synthesize_piper_wav(text: str, voice: str, speed: float) -> bytes:
    try:
        return _synthesize_piper_wav_inprocess(text, voice, speed)
    except Exception as inprocess_error:
        try:
            return _synthesize_piper_wav_subprocess(text, voice, speed)
        except Exception as subprocess_error:
            raise RuntimeError(
                "Piper synthesis failed in both in-process and subprocess modes. "
                f"In-process error: {inprocess_error}; subprocess error: {subprocess_error}"
            ) from subprocess_error


def _wav_to_float_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
    return audio, frame_rate


def warm_tts() -> None:
    """Validate Piper runtime and model paths before first speech request."""
    _ensure_piper_ready()
    try:
        _load_piper_voice()
    except Exception as e:
        # Subprocess fallback can still work if API loading fails.
        print(f"[TTS] In-process warmup unavailable, using subprocess fallback: {e}")


def speak(text: str, voice: str = "0", speed: float = 1.0) -> None:
    """
    Convert text to speech using Piper and play it through local speakers.
    """
    try:
        import sounddevice as sd

        wav_bytes = _synthesize_piper_wav(text, voice=voice, speed=speed)
        if not wav_bytes:
            return

        audio, sample_rate = _wav_to_float_audio(wav_bytes)
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except ImportError as ie:
        print(f"[TTS] Missing dependency: {ie}. Run: pip install sounddevice")
    except Exception as e:
        print(f"[TTS] Error: {e}")


async def speak_async(text: str, voice: str = "0", speed: float = 1.0) -> None:
    """
    Async wrapper for speak() for FastAPI/async contexts.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, speak, text, voice, speed)


def speak_sentence(text: str, voice: str = "0", speed: float = 1.0) -> None:
    """Speak a single sentence/chunk immediately."""
    speak(text, voice=voice, speed=speed)


def get_sentence_audio_bytes(text: str, voice: str = "0", speed: float = 1.0) -> bytes:
    """
    Generate TTS audio for a single sentence and return as WAV bytes.
    Designed for streaming sentence-by-sentence to the browser.
    """
    try:
        return _synthesize_piper_wav(text, voice=voice, speed=speed)
    except Exception as e:
        print(f"[TTS] get_sentence_audio_bytes error: {e}")
        return b""


def get_audio_bytes(text: str, voice: str = "0", speed: float = 1.0) -> bytes:
    """Generate TTS audio and return as WAV bytes."""
    try:
        return _synthesize_piper_wav(text, voice=voice, speed=speed)
    except Exception as e:
        print(f"[TTS] get_audio_bytes error: {e}")
        return b""
