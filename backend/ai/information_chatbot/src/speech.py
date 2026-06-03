"""
Speech I/O module — STELLAR-RAG v4.

STT  : PhoWhisper (VinAI) running in a SEPARATE SUBPROCESS.
       When the subprocess exits → 100% VRAM + RAM freed before Ollama runs.
       Never OOM from memory contention between STT and LLM.

TTS  : edge-tts  (Microsoft Edge TTS API, vi-VN-HoaiMyNeural, HTTP-only)
Rec  : sounddevice + soundfile  (16 kHz mono WAV)
Play : pygame.mixer  (cross-platform MP3)

Memory flow per turn:
  record_audio()          → temp WAV file
  speech_to_text()        → spawn subprocess → PhoWhisper load → transcribe
                          → subprocess exit  → VRAM + RAM 100% free
  agent.answer()          → Ollama has full resources 
  text_to_speech()        → edge-tts HTTP, no GPU required
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import warnings
from pathlib import Path

import numpy as np

# Suppress noisy warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Output path — single file, always overwritten
SPEECH_OUTPUT: Path = Path("storage") / "speech_output.mp3"

# Default model
DEFAULT_MODEL = "vinai/PhoWhisper-medium"

# Path to STT worker script
_WORKER_PATH: Path = Path(__file__).parent / "stt_worker.py"

# ─
# GPU detection (for display only — no model is loaded here)
# ─

def _device_label() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"GPU ({torch.cuda.get_device_name(0)}, fp16)"
    except ImportError:
        pass
    return "CPU (fp32)"

# ─
# Audio recording
# ─

def record_audio(samplerate: int = 16_000) -> str | None:
    """
    Record audio from the microphone until the user presses Enter.

    Returns the path to a temporary WAV file (deleted by speech_to_text()).
    Returns None on error.
    """
    try:
        import sounddevice as sd   # type: ignore
        import soundfile  as sf    # type: ignore
    except ImportError:
        print(
            "[STT]  sounddevice / soundfile chưa được cài đặt.\n"
            "       Chạy: pip install sounddevice soundfile"
        )
        return None

    print("\n[🎙]  Đang ghi âm…  Nhấn Enter để dừng.", flush=True)

    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def _callback(indata, _frames_count, _time_info, _status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            callback=_callback,
        ):
            input()          # block until user presses Enter
    except Exception as exc:
        print(f"[STT]  Lỗi ghi âm: {exc}")
        return None
    finally:
        stop_event.set()

    if not frames:
        print("[STT] WARNING:  Không ghi được âm thanh.")
        return None

    audio_data = np.concatenate(frames, axis=0)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        import soundfile as sf  # type: ignore
        sf.write(tmp.name, audio_data, samplerate)
    except Exception as exc:
        print(f"[STT]  Lỗi ghi file WAV: {exc}")
        return None

    return tmp.name

# ─
# STT — Speech to Text (PhoWhisper via subprocess)
# ─

def speech_to_text(wav_path: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Convert a WAV file to Vietnamese text using PhoWhisper.

    PhoWhisper runs in a separate subprocess:
    - Subprocess exit → VRAM + RAM 100% freed
    - Ollama receives full resources when called afterwards
    - WAV file is deleted inside the subprocess
    """
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(_WORKER_PATH), wav_path, model_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,   # maximum 2 minutes
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("[STT]  Timeout — nhận dạng quá 2 phút")
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        return ""
    except Exception as exc:
        print(f"[STT]  Không khởi động subprocess: {exc}")
        return ""

    # Display progress from stderr (loading weights, model info...)
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            line = line.strip()
            if line and not _should_suppress(line):
                print(f"[STT] {line}", flush=True)

    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"[STT]  Subprocess kết thúc với code {proc.returncode}")
        return ""

    try:
        data = json.loads(proc.stdout.strip())
        if data.get("error"):
            print(f"[STT]  {data['error']}")
            return ""
        return data.get("text", "")
    except json.JSONDecodeError:
        # Last stdout line should be JSON; if there is other output before it, take the last line
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        for line in reversed(lines):
            try:
                data = json.loads(line)
                return data.get("text", "")
            except json.JSONDecodeError:
                continue
        return ""

_NOISE_PATTERNS = (
    "403 Forbidden",
    "Discussions are disabled",
    "auto_conversion",
    "safetensors_conversion",
    "SuppressTokens",
    "clean_up_tokenization_spaces",
    "torch_dtype",
    "advisory",
    "HfHubHTTPError",
    "httpx.HTTPStatusError",
    "huggingface_hub",
    "Thread-auto",
    "UserWarning",
    "unauthenticated",           # "Warning: You are sending unauthenticated requests"
    "HF_TOKEN",                  # "Please set a HF_TOKEN"
    "rate limits",               # "enable higher rate limits"
    "FutureWarning",
    "DeprecationWarning",
)

def _should_suppress(line: str) -> bool:
    """Return True if the line is a warning or noise that should not be shown to the user."""
    return any(pat in line for pat in _NOISE_PATTERNS)

def unload_stt() -> None:
    """
    No-op — STT runs in a subprocess and frees its own memory on exit.
    Kept for backward compatibility with app.py.
    """
    pass

# ─
# TTS — Text to Speech
# ─

_VOICE_FEMALE = "vi-VN-HoaiMyNeural"   # natural female voice (default)
_VOICE_MALE   = "vi-VN-NamMinhNeural"  # natural male voice

def text_to_speech(
    text: str,
    output_path: Path | None = None,
    voice: str = _VOICE_FEMALE,
) -> Path | None:
    """
    Synthesise Vietnamese speech using edge-tts (HTTP, no GPU required).

    The MP3 file is written to output_path (default: storage/speech_output.mp3).
    Requires an internet connection.
    Returns None on failure.
    """
    if not text.strip():
        return None

    if output_path is None:
        output_path = SPEECH_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import edge_tts  # type: ignore
    except ImportError:
        print(
            "[TTS]  edge-tts chưa được cài đặt.\n"
            "       Chạy: pip install edge-tts"
        )
        return None

    async def _generate() -> None:
        comm = edge_tts.Communicate(text, voice)
        await comm.save(str(output_path))

    try:
        asyncio.run(_generate())
        return output_path
    except Exception as exc:
        print(f"[TTS]  Lỗi tổng hợp giọng nói: {exc}")
        print("       (edge-tts cần kết nối internet)")
        return None

# ─
# Audio playback
# ─

def play_audio(path: Path) -> None:
    """
    Phát file MP3/WAV bằng pygame.mixer. Chặn cho đến khi phát xong.
    """
    import time as _time

    try:
        import pygame  # type: ignore
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except ImportError:
        print(
            f"[TTS] WARNING:  Không thể phát âm thanh (pygame chưa cài).\n"
            f"       File đã lưu tại: {path}\n"
            f"       Chạy: pip install pygame"
        )
        return

    try:
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            _time.sleep(0.05)
    except Exception as exc:
        print(f"[TTS]  Lỗi phát audio: {exc}")
    finally:
        # Release file lock — pygame holds the handle even after playback ends.
        # Without unload, the next TTS call gets Permission denied when overwriting.
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

# ─
# High-level API (used by app.py)
# ─

def listen(model_name: str = DEFAULT_MODEL) -> str:
    """
    Record audio → STT (PhoWhisper subprocess) → return Vietnamese text.

    The subprocess exits after transcription → memory fully freed for Ollama.
    """
    wav = record_audio()
    if not wav:
        return ""

    print("[STT] Đang nhận dạng…", flush=True)
    text = speech_to_text(wav, model_name=model_name)

    if text:
        print(f"[Bạn nói]: {text}")
    else:
        print("[STT] WARNING:  Không nhận dạng được.")
    return text

def speak(text: str) -> None:
    """
    TTS → ghi đè speech_output.mp3 → phát âm thanh.
    """
    print("[TTS] Đang tổng hợp giọng nói…", end=" ", flush=True)
    path = text_to_speech(text)
    if path:
        print("")
        play_audio(path)
