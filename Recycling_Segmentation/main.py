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
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Recycling Segmentation API",
    description="재활용품 분류를 위한 Semantic Segmentation API",
    version="1.0.0"
)

# CORS 설정 (깔끔하게 최소화)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-dl-portfolio.onrender.com", "*"],
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# 모델 로드
MODEL_PATH = "./best_model"
model = None
processor = None

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 모델 로드"""
    global model, processor
    try:
        logger.info("🚀 모델 로드 중...")
        model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model.eval()
        
        # GPU 사용 가능시 GPU로 이동
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ GPU 모드로 모델 로드 완료!")
        else:
            logger.info("✅ CPU 모드로 모델 로드 완료!")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise

# 클래스명 및 색상
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

def create_visualization(image, mask):
    """시각화 함수 - overlay와 prediction 둘 다 생성"""
    overlay = image.copy().convert("RGB")
    pred = Image.new("RGB", image.size, (0, 0, 0))

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

        # Overlay 생성 (현실 배경 + 색상 오버레이)
        if class_id > 0:  # background 제외
            overlay_arr = np.array(overlay)
            mask_area = (mask == class_id)
            for c in range(3):
                overlay_arr[:, :, c][mask_area] = (
                    overlay_arr[:, :, c][mask_area] * 0.4 + 
                    color[c] * 0.6
                )
            overlay = Image.fromarray(overlay_arr)

        # Prediction 생성 (검정 배경 + 색상 채우기)
        pred_arr = np.array(pred)
        for c in range(3):
            pred_arr[:, :, c][mask == class_id] = color[c]
        pred = Image.fromarray(pred_arr)

        # 라벨 추가 (background 제외)
        if class_id > 0:
            yx = np.argwhere(mask == class_id)
            if len(yx) > 0:
                y_mean, x_mean = yx.mean(axis=0).astype(int)
                label = class_names[class_id]

                # 라벨 배경 및 텍스트 그리기
                for draw in [draw_pred, draw_overlay]:
                    draw.rectangle([(x_mean - 30, y_mean - 10), (x_mean + 30, y_mean + 10)], fill="black")
                    draw.text((x_mean - 20, y_mean - 8), label, fill="white", font=font)

    return pred, overlay

def create_overlay_only(image, mask):
    """실시간용 overlay만 생성 (속도 최적화)"""
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    mask = np.array(mask)

    for class_id in range(1, len(class_names)):  # background 제외
        if (mask == class_id).sum() == 0:
            continue

        color = class_colors_bright[class_id]
        
        # 색상 오버레이
        overlay_arr = np.array(overlay)
        mask_area = (mask == class_id)
        for c in range(3):
            overlay_arr[:, :, c][mask_area] = (
                overlay_arr[:, :, c][mask_area] * 0.4 + 
                color[c] * 0.6
            )
        overlay = Image.fromarray(overlay_arr)

        # 라벨 추가
        yx = np.argwhere(mask == class_id)
        if len(yx) > 0:
            y_mean, x_mean = yx.mean(axis=0).astype(int)
            label = class_names[class_id]
            
            draw.rectangle([(x_mean - 30, y_mean - 10), (x_mean + 30, y_mean + 10)], fill="black")
            draw.text((x_mean - 20, y_mean - 8), label, fill="white", font=font)

    return overlay

def predict_image(image_bytes, realtime=False):
    """이미지 예측 함수"""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 실시간 모드에서는 더 작은 크기로 처리하여 속도 향상
        if realtime:
            original_size = image.size
            if max(original_size) > 512:
                ratio = 512 / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image_for_processing = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                image_for_processing = image
        else:
            image_for_processing = image
            
        inputs = processor(images=image_for_processing, return_tensors="pt")

        # GPU 사용 가능시 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

        # 원본 크기로 리사이징
        original_size = image.size
        preds_resized = Image.fromarray(preds.astype(np.uint8)).resize(original_size, resample=Image.NEAREST)
        
        if realtime:
            # 실시간 모드: overlay만 반환
            overlay = create_overlay_only(image, preds_resized)
            return None, overlay
        else:
            # 일반 모드: 둘 다 반환
            pred, overlay = create_visualization(image, preds_resized)
            return pred, overlay
            
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """일반 예측 API 엔드포인트 (저장용)"""
    # 기본적인 파일 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 파일 크기 제한 (10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 10MB를 초과합니다")
    
    try:
        image_bytes = await file.read()
        pred_img, overlay_img = predict_image(image_bytes, realtime=False)

        # 이미지를 base64로 인코딩
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
    except Exception as e:
        logger.error(f"예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-realtime")
async def predict_realtime(file: UploadFile = File(...)):
    """실시간 예측 API 엔드포인트 (빠른 응답)"""
    # 기본적인 파일 검증
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    # 실시간용 파일 크기 제한 (5MB)
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 5MB를 초과합니다")
    
    try:
        start_time = time.time()
        image_bytes = await file.read()
        _, overlay_img = predict_image(image_bytes, realtime=True)

        # 이미지를 base64로 인코딩
        overlay_bytes = io.BytesIO()
        overlay_img.save(overlay_bytes, format="PNG", optimize=True)
        
        processing_time = time.time() - start_time
        logger.info(f"실시간 처리 시간: {processing_time:.3f}초")

        return {
            "success": True,
            "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
            "processing_time": round(processing_time, 3)
        }
    except Exception as e:
        logger.error(f"실시간 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "message": "Recycling Segmentation API"}

@app.head("/")
async def root_head():
    """헬스 체크 (HEAD 요청)"""
    return {}

@app.get("/classes")
async def get_classes():
    """클래스 정보 조회"""
    return {
        "classes": class_names,
        "colors": class_colors_bright
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
