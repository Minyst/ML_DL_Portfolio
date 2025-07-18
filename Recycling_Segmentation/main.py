from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import torch, numpy as np, io, os, base64, cv2
from scipy.ndimage import gaussian_filter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./best_model"
class_names = ["background", "can", "glass", "plastic", "styrofoam", "vinyl"]
class_colors = [
    (0, 0, 0), (0, 255, 255), (255, 255, 0),
    (255, 0, 0), (0, 128, 255), (255, 0, 128)
]

model = None
processor = None

def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except:
        return ImageFont.load_default()

def smooth_mask(mask, sigma=0.8):
    result = np.zeros_like(mask, dtype=np.float32)
    for i in range(len(class_names)):
        class_mask = (mask == i).astype(np.float32)
        if class_mask.sum() > 0:
            smoothed = gaussian_filter(class_mask, sigma=sigma)
            result[smoothed > 0.5] = i
    return result.astype(np.uint8)

def add_labels(image: Image.Image, mask: np.ndarray):
    draw = ImageDraw.Draw(image)
    font = load_font()
    for class_id in np.unique(mask):
        if class_id == 0 or class_id >= len(class_names):
            continue
        binary_mask = (mask == class_id).astype(np.uint8)
        num, labels = cv2.connectedComponents(binary_mask)
        for i in range(1, num):
            region = (labels == i)
            if region.sum() < 300: continue
            yx = np.argwhere(region)
            y, x = yx.mean(axis=0).astype(int)
            label = class_names[class_id]
            draw.rounded_rectangle([(x - 45, y - 18), (x + 45, y + 18)], radius=8, fill="black")
            draw.text((x - 35, y - 15), label, fill="white", font=font)
    return image

def create_overlay(image, mask):
    img = image.copy().convert("RGB")
    np_mask = np.array(mask)
    for class_id in range(1, len(class_names)):
        region = (np_mask == class_id)
        if region.sum() == 0: continue
        color = class_colors[class_id]
        arr = np.array(img)
        alpha = 0.3
        for c in range(3):
            arr[:, :, c][region] = arr[:, :, c][region] * (1 - alpha) + color[c] * alpha
        img = Image.fromarray(arr)
    return add_labels(img, np_mask)

def create_prediction(image, mask):
    pred = Image.new("RGB", image.size, (0, 0, 0))
    np_mask = np.array(mask)
    for class_id in range(1, len(class_names)):
        region = (np_mask == class_id)
        if region.sum() == 0: continue
        color = class_colors[class_id]
        arr = np.array(pred)
        for c in range(3):
            arr[:, :, c][region] = color[c]
        pred = Image.fromarray(arr)
    return add_labels(pred, np_mask)

@app.on_event("startup")
async def load_model():
    global model, processor
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

def run_inference(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = image.size
    image_resized = image.resize((512, 512))
    inputs = processor(images=image_resized, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
    resized_preds = cv2.resize(preds.astype(np.uint8), orig_size, interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(resized_preds)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    mask = run_inference(image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    overlay = create_overlay(image, mask)
    prediction = create_prediction(image, mask)
    overlay_io = io.BytesIO()
    pred_io = io.BytesIO()
    overlay.save(overlay_io, format="PNG")
    prediction.save(pred_io, format="PNG")
    return {
        "success": True,
        "overlay": base64.b64encode(overlay_io.getvalue()).decode("utf-8"),
        "prediction": base64.b64encode(pred_io.getvalue()).decode("utf-8"),
        "classes": class_names
    }

@app.get("/")
async def root():
    return {"message": "Segmentation API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")

