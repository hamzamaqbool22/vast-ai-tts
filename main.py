import io
import torch
import soundfile as sf

from fastapi import FastAPI, Form
from fastapi.responses import StreamingResponse

from transformers import AutoProcessor, AutoModel

app = FastAPI()

MODEL_PATH = "/workspace/tts_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load model once ----
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)

@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"ok": True}


# ---- TTS endpoint ----
@app.post("/tts")
async def tts(text: str = Form(...)):
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(**inputs)

    audio = output[0].cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, audio, samplerate=22050, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")
