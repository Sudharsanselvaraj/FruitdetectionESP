# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image
import numpy as np

# Initialize FastAPI
app = FastAPI(title="ESP32-CAM Fruit Detection API (Optimized)")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8n model for faster CPU inference
MODEL_PATH = "best_fruits_model.pt"  # Replace with YOLOv8n version if available
model = YOLO(MODEL_PATH)

@app.get("/")
async def home():
    return {"message": "Fruit Detection API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image from ESP32-CAM, resizes it, predicts fruits, and returns annotated image.
    """
    try:
        # Read uploaded image
        image = Image.open(file.file).convert("RGB")
        
        # Resize to smaller resolution for faster inference
        max_size = (320, 320)
        image.thumbnail(max_size, Image.ANTIALIAS)
        img_array = np.array(image)

        # Run YOLO prediction (no save, CPU optimized)
        results = model.predict(source=img_array, conf=0.4, save=False, show=False)

        # Annotate the image
        annotated_img = results[0].plot()  # returns numpy array
        annotated_pil = Image.fromarray(annotated_img)

        # Save image to bytes buffer
        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
