# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def root():
#     return {"status": "running"}


# @app.get("/health")
# def health():
#     return {"ok": True}
import io
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from qwen_tts import Qwen3TTSModel

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr),
    ],
    force=True,
)

logger = logging.getLogger("qwen_tts_api")
logger.setLevel(logging.INFO)
logger.propagate = False

# =========================================================
# Config
# =========================================================
MODEL_ID = os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MODEL_LOCAL_PATH = os.getenv(
    "QWEN_TTS_MODEL_PATH",
    "/workspace/models/Qwen3-TTS-12Hz-1.7B-Base",
)

PORT = int(os.getenv("PORT", "8000"))
MAX_REFERENCE_BYTES = int(os.getenv("MAX_REFERENCE_BYTES", str(25 * 1024 * 1024)))
MIN_REFERENCE_SECONDS = float(os.getenv("MIN_REFERENCE_SECONDS", "3.0"))
MAX_REFERENCE_SECONDS = float(os.getenv("MAX_REFERENCE_SECONDS", "15.0"))
TARGET_DECODE_SR = int(os.getenv("TARGET_DECODE_SR", "48000"))

ALLOWED_LANGUAGES = {
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
}

model_lock = Lock()


def log_event(request_id: str, stage: str, data=None):
    if data is None:
        logger.info(f"[{request_id}] {stage}")
    else:
        logger.info(f"[{request_id}] {stage} | {data}")


def _setup_torch():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _resolve_model_source() -> str:
    if os.path.isdir(MODEL_LOCAL_PATH):
        return MODEL_LOCAL_PATH
    return MODEL_ID


def _load_model_sync():
    _setup_torch()

    source = _resolve_model_source()
    dtype = _pick_dtype()

    log_event("SYSTEM", "MODEL_LOADING_START", {
        "source": source,
        "dtype": str(dtype),
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })

    last_error = None

    # Keep this stable. flash-attn is optional and not required.
    for attn_impl in ("sdpa", None):
        try:
            kwargs = {
                "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
                "dtype": dtype,
            }
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl

            log_event("SYSTEM", "MODEL_ATTEMPT", {"attn": attn_impl})

            tts_model = Qwen3TTSModel.from_pretrained(source, **kwargs)
            tts_model.model.eval()

            try:
                first_param = next(tts_model.model.parameters())
                log_event("SYSTEM", "MODEL_DEVICE", str(first_param.device))
            except Exception:
                pass

            log_event("SYSTEM", "MODEL_LOADED", {"attn": attn_impl})
            return tts_model, source, str(dtype)

        except Exception as exc:
            last_error = exc
            log_event("SYSTEM", "MODEL_LOAD_FAIL", {
                "attn": attn_impl,
                "error": str(exc),
            })

    raise RuntimeError(f"Failed to load Qwen3-TTS: {last_error}")


def _normalize_language(language: str) -> str:
    candidate = (language or "").strip()
    if not candidate:
        return "English"

    for allowed in ALLOWED_LANGUAGES:
        if allowed.lower() == candidate.lower():
            return allowed

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Unsupported language",
            "input": language,
            "allowed": sorted(ALLOWED_LANGUAGES),
        },
    )


def _fast_trim(wav: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if wav.size == 0:
        return wav

    idx = np.where(np.abs(wav) > threshold)[0]
    if len(idx) == 0:
        return wav[:0]

    start = max(int(idx[0]) - 1, 0)
    end = min(int(idx[-1]) + 2, len(wav))
    return wav[start:end]


def _decode_reference_audio(audio_bytes: bytes, filename: Optional[str]):
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty reference audio")

    # ffmpeg handles browser uploads like webm/opus, mp3, wav, m4a, etc.
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(TARGET_DECODE_SR),
        "-f",
        "wav",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not decode reference audio",
                "filename": filename,
                "error": err or "ffmpeg decode failed",
            },
        )

    try:
        wav, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Decoded audio could not be read",
                "error": str(exc),
            },
        )

    if isinstance(wav, np.ndarray) and wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav = np.asarray(wav, dtype=np.float32)

    if wav.size == 0:
        raise HTTPException(status_code=400, detail="Decoded reference audio is empty")

    wav = _fast_trim(wav, threshold=0.01)

    if wav.size == 0:
        raise HTTPException(status_code=400, detail="Reference audio contains only silence")

    duration = float(len(wav) / sr)

    if duration < MIN_REFERENCE_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Reference audio must be at least {MIN_REFERENCE_SECONDS:.0f} seconds after trimming",
        )

    if duration > MAX_REFERENCE_SECONDS:
        wav = wav[: int(MAX_REFERENCE_SECONDS * sr)]
        duration = float(len(wav) / sr)

    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        wav = wav / peak

    return wav, int(sr), duration


def _wav_bytes_from_float32(wav: np.ndarray, sr: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts_model = None
    app.state.model_ready = False
    app.state.model_error = None
    app.state.model_source = None
    app.state.model_dtype = None

    try:
        tts_model, source, dtype_str = await run_in_threadpool(_load_model_sync)
        app.state.tts_model = tts_model
        app.state.model_ready = True
        app.state.model_source = source
        app.state.model_dtype = dtype_str
    except Exception as exc:
        app.state.model_error = str(exc)
        logger.exception("Model init failed")

    yield


app = FastAPI(
    title="Qwen3-TTS Voice Clone API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "ok": True,
        "model_ready": bool(getattr(app.state, "model_ready", False)),
        "model_source": getattr(app.state, "model_source", None),
        "model_dtype": getattr(app.state, "model_dtype", None),
        "error": getattr(app.state, "model_error", None),
    }


@app.get("/health")
def health():
    return {
        "ok": bool(getattr(app.state, "model_ready", False)),
        "model_ready": bool(getattr(app.state, "model_ready", False)),
        "model_source": getattr(app.state, "model_source", None),
        "model_dtype": getattr(app.state, "model_dtype", None),
        "error": getattr(app.state, "model_error", None),
    }


@app.get("/debug/ping")
def debug_ping():
    logger.info("[SYSTEM] DEBUG_PING_HIT")
    return {"ok": True, "message": "ping received"}


@app.post("/v1/tts/clone")
async def clone_voice(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    language: str = Form("English"),
):
    request_id = f"req_{int(time.time() * 1000)}"
    start = time.time()

    log_event(request_id, "REQUEST_RECEIVED")

    try:
        if not getattr(app.state, "model_ready", False) or app.state.tts_model is None:
            log_event(request_id, "MODEL_NOT_READY", getattr(app.state, "model_error", None))
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Model not ready",
                    "error": getattr(app.state, "model_error", None),
                    "request_id": request_id,
                },
            )

        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required")

        language = _normalize_language(language)

        audio_bytes = await reference_audio.read()
        log_event(request_id, "AUDIO_RECEIVED", len(audio_bytes))

        if len(audio_bytes) > MAX_REFERENCE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"reference_audio exceeds max size of {MAX_REFERENCE_BYTES} bytes",
            )

        wav, sr, duration = await run_in_threadpool(
            _decode_reference_audio,
            audio_bytes,
            reference_audio.filename,
        )

        log_event(request_id, "AUDIO_READY", {"sr": sr, "dur": duration})

        ref_text = (reference_text or "").strip() or None
        x_vector_only_mode = ref_text is None

        def _run_model():
            log_event(request_id, "INFERENCE_START")
            with model_lock:
                output = app.state.tts_model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=(wav, sr),
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    non_streaming_mode=True,
                )
            log_event(request_id, "INFERENCE_DONE")
            return output

        wavs, out_sr = await run_in_threadpool(_run_model)

        if not wavs:
            raise HTTPException(status_code=500, detail="Empty model output")

        out = np.asarray(wavs[0], dtype=np.float32)
        audio = _wav_bytes_from_float32(out, int(out_sr))

        total = time.time() - start
        log_event(request_id, "REQUEST_DONE", f"{total:.2f}s")

        headers = {
            "Content-Disposition": 'inline; filename="voice.wav"',
            "X-Request-ID": request_id,
            "X-Reference-Duration-Seconds": f"{duration:.2f}",
            "X-Total-Time-Seconds": f"{total:.2f}",
        }
        if x_vector_only_mode:
            headers["X-Warning"] = "reference_text was missing; x_vector_only_mode=true used"

        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/wav",
            headers=headers,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"[{request_id}] ERROR")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "TTS generation failed",
                "error": str(exc),
                "request_id": request_id,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        workers=1,
        reload=False,
    )
