# main.py (FastAPI 서버 - 개선된 Overlay/Predict 시각화)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import io
import os
import uvicorn
import base64
import time
import cv2
from scipy.ndimage import gaussian_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./best_model"
model = None
processor = None

class_names = ["background", "can", "glass", "plastic", "styrofoam", "vinyl", "others"]
class_colors_bright = [
    (0, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0),
    (0, 128, 255), (255, 0, 128), (160, 160, 160)
]

LABEL_AREA_THRESHOLD = 1000
LABEL_PER_CLASS_LIMIT = 1


def load_font():
    try:
        return ImageFont.truetype("assets/fonts/Pretendard-Medium.otf", 18)
    except:
        return ImageFont.load_default()

def smooth_mask(mask, sigma=0.8):
    smoothed = np.zeros_like(mask, dtype=np.float32)
    for class_id in range(len(class_names)):
        class_mask = (mask == class_id).astype(np.float32)
        if class_mask.sum() > 0:
            smoothed_class = gaussian_filter(class_mask, sigma=sigma)
            smoothed[smoothed_class > 0.5] = class_id
    return smoothed.astype(np.uint8)

def add_center_labels(image: Image.Image, mask: np.ndarray):
    draw = ImageDraw.Draw(image)
    font = load_font()
    H, W = mask.shape
    cx_min, cx_max = W // 3, W * 2 // 3
    cy_min, cy_max = H // 3, H * 2 // 3
    for class_id in np.unique(mask):
        if class_id == 0 or class_id >= len(class_names):
            continue
        class_mask = (mask == class_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(class_mask)
        components = []
        for label_id in range(1, num_labels):
            component_mask = (labels == label_id)
            area = component_mask.sum()
            if area >= LABEL_AREA_THRESHOLD:
                yx = np.argwhere(component_mask)
                y_mean, x_mean = yx.mean(axis=0)
                if cx_min < x_mean < cx_max and cy_min < y_mean < cy_max:
                    components.append((area, component_mask))
        components.sort(reverse=True, key=lambda x: x[0])
        components = components[:LABEL_PER_CLASS_LIMIT]
        for _, component_mask in components:
            yx = np.argwhere(component_mask)
            y_min, x_min = yx.min(axis=0)
            y_max, x_max = yx.max(axis=0)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            label = class_names[class_id]
            draw.rounded_rectangle(
                [(x_center - 50, y_center - 18), (x_center + 50, y_center + 18)],
                radius=6, fill="black"
            )
            draw.text((x_center - 35, y_center - 12), label, fill="white", font=font)
    return image

def create_visualization(image, mask):
    mask_smooth = smooth_mask(mask)
    overlay = image.copy().convert("RGB")
    pred = Image.new("RGB", image.size, (0, 0, 0))
    mask_np = np.array(mask_smooth)
    for class_id in range(len(class_names)):
        class_mask = (mask_np == class_id)
        if class_mask.sum() == 0:
            continue
        color = class_colors_bright[class_id]
        overlay_arr = np.array(overlay)
        pred_arr = np.array(pred)
        alpha = 0.35
        for c in range(3):
            overlay_arr[:, :, c][class_mask] = overlay_arr[:, :, c][class_mask] * (1 - alpha) + color[c] * alpha
            pred_arr[:, :, c][class_mask] = color[c]
        overlay = Image.fromarray(overlay_arr)
        pred = Image.fromarray(pred_arr)
    pred = add_center_labels(pred, mask_np)
    overlay = add_center_labels(overlay, mask_np)
    return pred, overlay

def create_overlay_only(image, mask):
    overlay = image.copy().convert("RGB")
    mask = np.array(mask)
    for class_id in range(1, len(class_names)):
        class_mask = (mask == class_id)
        if class_mask.sum() == 0:
            continue
        color = class_colors_bright[class_id]
        overlay_arr = np.array(overlay)
        alpha = 0.3
        for c in range(3):
            overlay_arr[:, :, c][class_mask] = overlay_arr[:, :, c][class_mask] * (1 - alpha) + color[c] * alpha
        overlay = Image.fromarray(overlay_arr)
    overlay = add_center_labels(overlay, mask)
    return overlay

def predict_image(image_bytes, realtime=False):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    image_resized = image.resize((512, 512))
    inputs = processor(images=image_resized, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    preds_resized = cv2.resize(preds.astype(np.uint8), original_size, interpolation=cv2.INTER_CUBIC)
    preds_resized = Image.fromarray(preds_resized)
    if realtime:
        overlay = create_overlay_only(image, preds_resized)
        return None, overlay
    else:
        pred, overlay = create_visualization(image, preds_resized)
        return pred, overlay

@app.on_event("startup")
async def load_model():
    global model, processor
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info("Model loaded.")

@app.get("/")
async def root():
    return {"message": "Segmentation API is running. Use /predict or /predict-realtime."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred_img, overlay_img = predict_image(image_bytes, realtime=False)
    pred_bytes = io.BytesIO()
    overlay_bytes = io.BytesIO()
    pred_img.save(pred_bytes, format="PNG")
    overlay_img.save(overlay_bytes, format="PNG")
    return {
        "success": True,
        "prediction": base64.b64encode(pred_bytes.getvalue()).decode("utf-8"),
        "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
        "classes": class_names
    }

@app.post("/predict-realtime")
async def predict_realtime(file: UploadFile = File(...)):
    image_bytes = await file.read()
    _, overlay_img = predict_image(image_bytes, realtime=True)
    overlay_bytes = io.BytesIO()
    overlay_img.save(overlay_bytes, format="PNG")
    return {
        "success": True,
        "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8")
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
