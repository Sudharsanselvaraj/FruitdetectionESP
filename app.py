# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
from datetime import datetime

# Initialize FastAPI
app = FastAPI(title="ESP32-CAM Fruit Detection API")

# Allow CORS for all origins (so your frontend or ESP32 can call it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
MODEL_PATH = "best_fruits.pt"  # Make sure this is in your repo
model = YOLO(MODEL_PATH)

# Create a folder to save uploaded images and predictions
UPLOAD_DIR = "uploads"
PRED_DIR = "runs/detect/predict"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


@app.get("/")
async def home():
    return {"message": "Fruit Detection API is running!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns the annotated image with detected fruits.
    """
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLOv8 prediction
        results = model.predict(source=file_path, conf=0.4, save=True, show=False)

        # YOLO saves output in runs/detect/predict by default
        # Grab the latest saved image
        pred_image_path = results[0].plot(save=False)  # returns numpy array (optional)
        save_path = os.path.join(PRED_DIR, f"pred_{timestamp}_{file.filename}")
        results[0].save(save_path)  # save annotated image

        return FileResponse(save_path)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
