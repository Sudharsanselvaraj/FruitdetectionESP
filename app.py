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

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
MODEL_PATH = "best_fruits_model.pt"
model = YOLO(MODEL_PATH)

# Create directories
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def home():
    return {"message": "Fruit Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(UPLOAD_DIR, f"{timestamp}_{file.filename}")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLOv8 inference and save annotated output
        results = model.predict(
            source=input_path,
            conf=0.4,
            save=True,
            project="runs/detect",
            name="predict",
            exist_ok=True
        )

        # Find the latest predict folder (e.g. runs/detect/predict3)
        predict_dirs = sorted(glob.glob("runs/detect/predict*"), key=os.path.getctime)
        if not predict_dirs:
            return JSONResponse(status_code=500, content={"error": "No prediction folder found."})

        latest_dir = predict_dirs[-1]

        # Find the annotated image inside that folder
        image_files = glob.glob(os.path.join(latest_dir, "*.jpg"))
        if not image_files:
            return JSONResponse(status_code=500, content={"error": "No annotated image found."})

        latest_image = max(image_files, key=os.path.getctime)

        # ✅ Return image directly
        return FileResponse(latest_image, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
