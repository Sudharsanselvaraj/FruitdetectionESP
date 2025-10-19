from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
from datetime import datetime
import glob

# Initialize FastAPI
app = FastAPI(title="ESP32-CAM Fruit Detection API")

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
PRED_DIR = "runs/detect/predict"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

@app.get("/")
async def home():
    return {"message": "Fruit Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLOv8 prediction and save results
        results = model.predict(source=file_path, conf=0.4, save=True, project="runs/detect", name="predict")

        # Get the latest saved image
        saved_images = glob.glob(os.path.join(PRED_DIR, "*"))
        latest_image = max(saved_images, key=os.path.getctime)

        return FileResponse(latest_image)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
