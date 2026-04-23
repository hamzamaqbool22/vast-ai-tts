import io
import torch
import soundfile as sf

from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse
from qwen_tts import Qwen3TTSModel

app = FastAPI()

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading TTS model...")

model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

print("Model loaded.")

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/tts")
def tts(text: str = Form(...)):

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="auto"
    )

    buf = io.BytesIO()
    sf.write(buf, wavs[0], sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
