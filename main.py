from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"ok": True}
