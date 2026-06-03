"""
STT Worker — runs in a separate subprocess.

When this subprocess exits, 100% of PhoWhisper's VRAM + RAM is freed
before Ollama (a different process) runs — guarantees no OOM.

Usage:
    python stt_worker.py <wav_path> [<model_name>]

Stdout: JSON  {"text": "...", "error": null}
"""
import sys
import os
import json
import threading

# Suppress all warnings/noise before any imports
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

# Silence 403 HuggingFace discussions thread exception completely
threading.excepthook = lambda _args: None

def _patch_safetensors():
    """Disable background thread auto-conversion (causes 403 noise)."""
    try:
        import transformers.safetensors_conversion as _sc
        _sc.auto_conversion = lambda *a, **kw: None
    except Exception:
        pass

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"text": "", "error": "Missing wav_path argument"}))
        sys.exit(1)

    wav_path   = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "vinai/PhoWhisper-medium"

    _patch_safetensors()

    try:
        import torch
        from transformers import pipeline as hf_pipeline  # type: ignore
        import soundfile as sf  # type: ignore
        import transformers
        transformers.logging.set_verbosity_error()

        # Select device
        if torch.cuda.is_available():
            device = "cuda:0"
            dtype  = torch.float16
        else:
            device = "cpu"
            dtype  = torch.float32

        # Load ASR pipeline
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=dtype,
            device=device,
            generate_kwargs={"language": "vi", "task": "transcribe"},
        )

        # Read WAV file
        audio, sample_rate = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)   # convert stereo to mono

        # Run transcription
        result = pipe({"raw": audio, "sampling_rate": sample_rate})
        text   = result.get("text", "").strip()   # type: ignore[index]

        print(json.dumps({"text": text, "error": None}, ensure_ascii=False))

    except Exception as exc:
        print(json.dumps({"text": "", "error": str(exc)}, ensure_ascii=False))

    finally:
        # Delete temporary WAV file
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    # When main() returns, subprocess exits → OS reclaims all VRAM + RAM

if __name__ == "__main__":
    main()
