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

app = FastAPI()

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-dl-portfolio.onrender.com/predict"],  
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ✅ 모델 로드
MODEL_PATH = "./best_model"

print("🚀 모델 로드 중...")
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model.eval()
print("✅ 모델 로드 완료!")

# ✅ 클래스명 및 색상
class_names = [
    "background", "can", "glass",
    "paper", "plastic", "styrofoam", "vinyl"
]
class_colors_bright = [
    (0, 0, 0),        # background - 검정
    (0, 255, 255),    # can - 청록
    (255, 255, 0),    # glass - 노랑
    (128, 255, 0),    # paper - 연두
    (255, 0, 0),      # plastic - 빨강
    (0, 128, 255),    # styrofoam - 파랑
    (255, 0, 128)     # vinyl - 분홍
]

# ✅ 시각화 함수
def create_visualization(image, mask):
    overlay = image.copy().convert("RGB")  # 현실 배경 유지
    pred = Image.new("RGB", image.size, (0, 0, 0))  # 검정 배경

    draw_overlay = ImageDraw.Draw(overlay)
    draw_pred = ImageDraw.Draw(pred)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    mask = np.array(mask)

    for class_id in range(len(class_names)):
        if (mask == class_id).sum() == 0:
            continue

        color = class_colors_bright[class_id]

        # Overlay: 현실 배경 + 색상으로 완전히 덮기
        overlay_arr = np.array(overlay)
        for c in range(3):  # R, G, B 채널별로 색상 적용
            overlay_arr[:, :, c][mask == class_id] = color[c]
        overlay = Image.fromarray(overlay_arr)

        # Prediction: 검정 배경 + 색상으로 완전히 덮기
        pred_arr = np.array(pred)
        for c in range(3):
            pred_arr[:, :, c][mask == class_id] = color[c]
        pred = Image.fromarray(pred_arr)

        # 라벨 중앙 위치 계산
        yx = np.argwhere(mask == class_id)
        if len(yx) > 0:
            y_mean, x_mean = yx.mean(axis=0).astype(int)
            label = class_names[class_id]

            # 검은 박스 안에 흰색 라벨 (Prediction)
            draw_pred.rectangle([(x_mean - 30, y_mean - 10), (x_mean + 30, y_mean + 10)], fill="black")
            draw_pred.text((x_mean - 20, y_mean - 8), label, fill="white", font=font)

            # 검은 박스 안에 흰색 라벨 (Overlay)
            draw_overlay.rectangle([(x_mean - 30, y_mean - 10), (x_mean + 30, y_mean + 10)], fill="black")
            draw_overlay.text((x_mean - 20, y_mean - 8), label, fill="white", font=font)

    return pred, overlay

# ✅ 예측 + 시각화
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    pred_img, overlay_img = create_visualization(image, preds)
    return pred_img, overlay_img

# ✅ API 엔드포인트
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pred_img, overlay_img = predict_image(image_bytes)

    # PNG → Base64 변환
    pred_bytes = io.BytesIO()
    overlay_bytes = io.BytesIO()
    pred_img.save(pred_bytes, format="PNG")
    overlay_img.save(overlay_bytes, format="PNG")

    pred_base64 = base64.b64encode(pred_bytes.getvalue()).decode("utf-8")
    overlay_base64 = base64.b64encode(overlay_bytes.getvalue()).decode("utf-8")

    return {
        "prediction": pred_base64,
        "overlay": overlay_base64
    }

# ✅ Render 배포용 포트 설정
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
