from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import io
import os
import uvicorn
import base64
import cv2

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
class_colors = [
    (0, 0, 0), (0, 255, 255), (255, 255, 0),
    (255, 0, 0), (0, 128, 255), (255, 0, 128), (128, 128, 128)
]

def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except:
        return ImageFont.load_default()

def add_labels(img_pil, mask):
    draw = ImageDraw.Draw(img_pil)
    font = load_font()
    for cid in np.unique(mask):
        if cid == 0 or cid >= len(class_names):
            continue
        m = (mask == cid).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(m)
        max_area = 0
        center = None
        for i in range(1, num_labels):
            coords = np.column_stack(np.where(labels == i))
            area = len(coords)
            if area > max_area:
                max_area = area
                center = coords.mean(axis=0).astype(int)
        if center is not None:
            x, y = center[1], center[0]
            label = class_names[cid]
            draw.rounded_rectangle([(x - 45, y - 18), (x + 45, y + 18)], radius=10, fill="black")
            draw.text((x - 35, y - 15), label, fill="white", font=font)
    return img_pil

def predict(image_bytes, realtime=False):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    image_resized = image.resize((512, 512))
    inputs = processor(images=image_resized, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    preds_resized = cv2.resize(preds.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)

    overlay = image.copy().convert("RGBA")
    pred_img = Image.new("RGB", image.size, (0, 0, 0))
    for cid in range(1, len(class_names)):
        mask = (preds_resized == cid)
        if mask.sum() == 0:
            continue
        color = class_colors[cid]
        alpha = 100
        overlay_np = np.array(overlay)
        for c in range(3):
            overlay_np[:, :, c][mask] = color[c]
        overlay_np[:, :, 3][mask] = alpha
        overlay = Image.fromarray(overlay_np)

        pred_arr = np.array(pred_img)
        for c in range(3):
            pred_arr[:, :, c][mask] = color[c]
        pred_img = Image.fromarray(pred_arr)

    overlay = add_labels(overlay, preds_resized)
    pred_img = add_labels(pred_img, preds_resized)

    return pred_img, overlay

@app.on_event("startup")
async def load():
    global model, processor
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model.eval().cuda() if torch.cuda.is_available() else model.eval()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pred_img, overlay = predict(img_bytes, realtime=False)
    pred_buf = io.BytesIO()
    overlay_buf = io.BytesIO()
    pred_img.save(pred_buf, format="PNG")
    overlay.save(overlay_buf, format="PNG")
    return {
        "success": True,
        "prediction": base64.b64encode(pred_buf.getvalue()).decode(),
        "overlay": base64.b64encode(overlay_buf.getvalue()).decode()
    }

@app.post("/predict-realtime")
async def predict_realtime(file: UploadFile = File(...)):
    img_bytes = await file.read()
    _, overlay = predict(img_bytes, realtime=True)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return {
        "success": True,
        "overlay": base64.b64encode(buf.getvalue()).decode()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
