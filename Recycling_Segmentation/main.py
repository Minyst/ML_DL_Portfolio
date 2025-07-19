from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch, numpy as np, io, os, base64, cv2
from scipy.ndimage import gaussian_filter
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "./best_model"
class_names = ["background", "can", "glass", "paper", "plastic", "styrofoam", "vinyl"]
class_colors = [
    (0, 0, 0),        # background - 검은색
    (0, 255, 255),    # can - 밝은 청록색
    (255, 255, 0),    # glass - 밝은 노란색
    (128, 255, 0),    # paper - 연두색
    (255, 0, 0),      # plastic - 밝은 빨간색
    (0, 128, 255),    # styrofoam - 밝은 파란색
    (255, 0, 128)     # vinyl - 밝은 분홍색
]

model = None
processor = None

def load_font():
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except:
        return ImageFont.load_default()

def smooth_mask_advanced(mask, sigma=1.5):
    """더 부드러운 마스크 생성"""
    result = np.zeros_like(mask, dtype=np.float32)
    
    # 각 클래스별로 처리
    for i in range(len(class_names)):
        if i == 0:  # background 스킵
            continue
            
        class_mask = (mask == i).astype(np.float32)
        if class_mask.sum() > 0:
            # 가우시안 블러 적용
            smoothed = gaussian_filter(class_mask, sigma=sigma)
            
            # 모폴로지 연산으로 더 부드럽게
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            class_mask_uint8 = (class_mask * 255).astype(np.uint8)
            
            # 닫힘 연산 (구멍 메우기)
            closed = cv2.morphologyEx(class_mask_uint8, cv2.MORPH_CLOSE, kernel)
            
            # 열림 연산 (노이즈 제거)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # 가우시안 블러로 최종 부드럽게
            final_smooth = cv2.GaussianBlur(opened, (9, 9), 2.0)
            
            # 임계값 적용
            result[final_smooth > 127] = i
    
    return result.astype(np.uint8)

def create_smooth_edge_mask(mask, edge_blur=3):
    """가장자리를 부드럽게 만드는 함수"""
    result = mask.copy()
    
    for class_id in range(1, len(class_names)):
        class_mask = (mask == class_id).astype(np.uint8) * 255
        if class_mask.sum() == 0:
            continue
            
        # 가장자리 감지
        edges = cv2.Canny(class_mask, 50, 150)
        
        # 가장자리 주변을 블러 처리
        kernel = np.ones((edge_blur*2+1, edge_blur*2+1), np.uint8)
        edge_region = cv2.dilate(edges, kernel, iterations=1)
        
        # 해당 영역에 가우시안 블러 적용
        blurred_mask = cv2.GaussianBlur(class_mask, (edge_blur*2+1, edge_blur*2+1), edge_blur/2)
        
        # 원본과 블러된 마스크를 블렌딩
        blend_factor = edge_region.astype(np.float32) / 255.0
        final_mask = class_mask.astype(np.float32) * (1 - blend_factor) + blurred_mask.astype(np.float32) * blend_factor
        
        result[final_mask > 127] = class_id
    
    return result

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
            if region.sum() < 500:  # 최소 영역 크기 증가
                continue
                
            yx = np.argwhere(region)
            y, x = yx.mean(axis=0).astype(int)
            
            label = class_names[class_id]
            
            # 더 부드러운 라벨 배경
            draw.rounded_rectangle(
                [(x - 50, y - 20), (x + 50, y + 20)], 
                radius=12,  # 더 둥근 모서리
                fill="black"
            )
            draw.text((x - 40, y - 12), label, fill="white", font=font)
    
    return image

def create_overlay(image, mask):
    img = image.copy().convert("RGB")
    
    # 부드러운 마스크 적용
    smooth_mask = smooth_mask_advanced(np.array(mask))
    smooth_mask = create_smooth_edge_mask(smooth_mask)
    
    for class_id in range(1, len(class_names)):
        region = (smooth_mask == class_id)
        if region.sum() == 0:
            continue
            
        color = class_colors[class_id]
        arr = np.array(img)
        alpha = 0.4  # 투명도 약간 증가
        
        for c in range(3):
            arr[:, :, c][region] = arr[:, :, c][region] * (1 - alpha) + color[c] * alpha
        
        img = Image.fromarray(arr)
    
    # 가장자리 블러 효과 추가
    img = img.filter(ImageFilter.SMOOTH_MORE)
    
    return add_labels(img, smooth_mask)

def create_prediction(image, mask):
    pred = Image.new("RGB", image.size, (0, 0, 0))
    
    # 부드러운 마스크 적용
    smooth_mask = smooth_mask_advanced(np.array(mask))
    smooth_mask = create_smooth_edge_mask(smooth_mask)
    
    for class_id in range(1, len(class_names)):
        region = (smooth_mask == class_id)
        if region.sum() == 0:
            continue
            
        color = class_colors[class_id]
        arr = np.array(pred)
        
        for c in range(3):
            arr[:, :, c][region] = color[c]
        
        pred = Image.fromarray(arr)
    
    # 전체적으로 부드럽게 처리
    pred = pred.filter(ImageFilter.SMOOTH)
    
    return add_labels(pred, smooth_mask)

@app.on_event("startup")
async def load_model():
    global model, processor
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, use_fast=True)
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
    return {"message": "Enhanced Segmentation API is running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
