from fastapi import FastAPI
from .routes import predict

app = FastAPI(title="Dynamic Inference API")
app.include_router(predict.router, prefix="")

@app.get("/")
def read_root():
    return {"status": "ok", "api": "inference"}
