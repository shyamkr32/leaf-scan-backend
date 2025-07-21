from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.model.yolov5 import predict_image
from app.data.remedies import remedies
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_file_path)
    os.remove(temp_file_path)

    if result is None:
        return {"error": "Try uploading different image"}

    class_id, confidence = result
    remedy_info = remedies.get(class_id)

    if remedy_info:
        return {
            "class_id": class_id,
            "disease": remedy_info["name"],
            "confidence": f"{confidence * 100:.2f}%",
            "remedy": remedy_info["remedy"],
            "cause": remedy_info["cause"]
        }
    else:
        return {"error": "Unknown disease detected."}
