from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
from datetime import datetime
import glob
import io

# Initialize FastAPI
app = FastAPI(title="ESP32-CAM Fruit Detection API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
MODEL_PATH = "best_fruits_model.pt"
model = YOLO(MODEL_PATH)

# Create folders
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def home():
    return {"message": "Fruit Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLO prediction and save annotated image
        results = model.predict(
            source=file_path,
            conf=0.4,
            save=True,
            project="runs/detect",
            name="predict",
            exist_ok=True  # ensures YOLO doesn't create exp, exp2 folders
        )

        # The YOLO result object contains .save() info. Grab first result
        annotated_image_path = results[0].save()  # returns saved image path

        # Read image as bytes and return
        with open(annotated_image_path, "rb") as f_img:
            buf = io.BytesIO(f_img.read())
            buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
