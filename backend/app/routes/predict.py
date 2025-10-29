from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
from ..services.inference_service import do_inference
import io

router = APIRouter()

@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    result = do_inference(image)
    return result
