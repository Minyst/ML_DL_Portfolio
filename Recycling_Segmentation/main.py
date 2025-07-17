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
import cv2
from scipy.ndimage import gaussian_filter

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Recycling Segmentation API",
    description="재활용품 분류를 위한 Semantic Segmentation API",
    version="1.0.0"
)

# CORS 설정 (모바일 앱 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모바일 앱용 임시 설정
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Origin"],
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

def smooth_mask(mask, sigma=1.0):
    """마스크 부드럽게 처리"""
    smoothed = np.zeros_like(mask, dtype=np.float32)
    for class_id in range(len(class_names)):
        class_mask = (mask == class_id).astype(np.float32)
        if class_mask.sum() > 0:
            smoothed_class = gaussian_filter(class_mask, sigma=sigma)
            smoothed[smoothed_class > 0.5] = class_id
    return smoothed.astype(np.uint8)

def add_contours(image, mask):
    """테두리 추가"""
    result = np.array(image)
    
    for class_id in range(1, len(class_names)):
        if (mask == class_id).sum() == 0:
            continue
            
        # 클래스별 마스크 생성
        class_mask = (mask == class_id).astype(np.uint8) * 255
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 테두리 그리기 (더 굵게)
        color = class_colors_bright[class_id]
        cv2.drawContours(result, contours, -1, color, 3)
        
        # 내부 테두리 (더 부드럽게)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
    
    return Image.fromarray(result)

def create_visualization(image, mask):
    """시각화 함수 - overlay와 prediction 둘 다 생성 (개선된 버전)"""
    # 마스크 부드럽게 처리
    mask_smooth = smooth_mask(mask, sigma=0.8)
    
    overlay = image.copy().convert("RGB")
    pred = Image.new("RGB", image.size, (0, 0, 0))

    draw_overlay = ImageDraw.Draw(overlay)
    draw_pred = ImageDraw.Draw(pred)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 18)  # Windows
        except:
            font = ImageFont.load_default()

    mask_smooth = np.array(mask_smooth)

    for class_id in range(len(class_names)):
        if (mask_smooth == class_id).sum() == 0:
            continue

        color = class_colors_bright[class_id]

        # Overlay 생성 (현실 배경 + 색상 오버레이) - 더 자연스럽게
        if class_id > 0:  # background 제외
            overlay_arr = np.array(overlay)
            mask_area = (mask_smooth == class_id)
            
            # 알파 블렌딩 개선 (더 투명하게)
            alpha = 0.3
            for c in range(3):
                overlay_arr[:, :, c][mask_area] = (
                    overlay_arr[:, :, c][mask_area] * (1 - alpha) + 
                    color[c] * alpha
                )
            overlay = Image.fromarray(overlay_arr)

        # Prediction 생성 (검정 배경 + 색상 채우기)
        pred_arr = np.array(pred)
        for c in range(3):
            pred_arr[:, :, c][mask_smooth == class_id] = color[c]
        pred = Image.fromarray(pred_arr)

        # 라벨 추가 (background 제외) - 더 큰 글자
        if class_id > 0:
            yx = np.argwhere(mask_smooth == class_id)
            if len(yx) > 0:
                y_mean, x_mean = yx.mean(axis=0).astype(int)
                label = class_names[class_id]

                # 라벨 배경 및 텍스트 그리기 (더 큰 박스)
                for draw in [draw_pred, draw_overlay]:
                    draw.rectangle([(x_mean - 40, y_mean - 15), (x_mean + 40, y_mean + 15)], fill="black")
                    draw.text((x_mean - 30, y_mean - 12), label, fill="white", font=font)

    # 테두리 추가 (prediction에만)
    pred = add_contours(pred, mask_smooth)
    
    return pred, overlay

def create_overlay_only(image, mask):
    """실시간용 overlay만 생성 (속도 최적화 + 품질 개선)"""
    # 실시간에서는 가벼운 부드럽게 처리
    mask_smooth = smooth_mask(mask, sigma=0.5)
    
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 16)  # Windows
        except:
            font = ImageFont.load_default()

    mask_smooth = np.array(mask_smooth)

    for class_id in range(1, len(class_names)):  # background 제외
        if (mask_smooth == class_id).sum() == 0:
            continue

        color = class_colors_bright[class_id]
        
        # 색상 오버레이 (더 자연스럽게)
        overlay_arr = np.array(overlay)
        mask_area = (mask_smooth == class_id)
        
        # 알파 블렌딩 개선
        alpha = 0.25  # 실시간에서는 더 투명하게
        for c in range(3):
            overlay_arr[:, :, c][mask_area] = (
                overlay_arr[:, :, c][mask_area] * (1 - alpha) + 
                color[c] * alpha
            )
        overlay = Image.fromarray(overlay_arr)

        # 라벨 추가
        yx = np.argwhere(mask_smooth == class_id)
        if len(yx) > 0:
            y_mean, x_mean = yx.mean(axis=0).astype(int)
            label = class_names[class_id]
            
            # 라벨 배경 및 텍스트 그리기
            draw.rectangle([(x_mean - 35, y_mean - 12), (x_mean + 35, y_mean + 12)], fill="black")
            draw.text((x_mean - 25, y_mean - 10), label, fill="white", font=font)

    # 가벼운 테두리 추가
    overlay = add_contours(overlay, mask_smooth)
    
    return overlay

def predict_image(image_bytes, realtime=False):
    """이미지 예측 함수 (개선된 버전)"""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        
        logger.info(f"입력 이미지 크기: {original_size}")
        
        # DeepLabV3 + MobileViT 최적화된 전처리
        if realtime:
            # 실시간: 품질과 속도 균형
            target_size = 512
            if max(original_size) > target_size:
                ratio = target_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image_for_processing = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                image_for_processing = image
        else:
            # 일반: 고품질
            target_size = 768  # 더 높은 해상도
            if max(original_size) > target_size:
                ratio = target_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image_for_processing = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                image_for_processing = image
        
        logger.info(f"처리 이미지 크기: {image_for_processing.size}")
        
        # MobileViT 최적화된 전처리
        inputs = processor(
            images=image_for_processing, 
            return_tensors="pt",
            do_resize=True,
            size={"height": 512, "width": 512},  # MobileViT 최적 크기
            do_normalize=True
        )

        # GPU 사용 가능시 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            
            # 소프트맥스 적용 (더 부드러운 경계)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

        logger.info(f"모델 출력 크기: {preds.shape}")

        # 원본 크기로 부드럽게 리사이징
        if preds.shape != (original_size[1], original_size[0]):
            # OpenCV 사용 (더 부드러운 결과)
            preds_resized = cv2.resize(
                preds.astype(np.uint8), 
                original_size, 
                interpolation=cv2.INTER_NEAREST
            )
            preds_resized = Image.fromarray(preds_resized)
        else:
            preds_resized = Image.fromarray(preds.astype(np.uint8))
        
        logger.info(f"최종 출력 크기: {preds_resized.size}")
        
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
        start_time = time.time()
        image_bytes = await file.read()
        pred_img, overlay_img = predict_image(image_bytes, realtime=False)

        # 이미지를 base64로 인코딩
        pred_bytes = io.BytesIO()
        overlay_bytes = io.BytesIO()
        pred_img.save(pred_bytes, format="PNG", optimize=True)
        overlay_img.save(overlay_bytes, format="PNG", optimize=True)
        
        processing_time = time.time() - start_time
        logger.info(f"일반 처리 시간: {processing_time:.3f}초")

        return {
            "success": True,
            "prediction": base64.b64encode(pred_bytes.getvalue()).decode("utf-8"),
            "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
            "classes": class_names,
            "processing_time": round(processing_time, 3)
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
    
    # 실시간용 파일 크기 제한 (8MB)
    if file.size and file.size > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="파일 크기가 8MB를 초과합니다")
    
    try:
        start_time = time.time()
        image_bytes = await file.read()
        _, overlay_img = predict_image(image_bytes, realtime=True)

        # 이미지를 base64로 인코딩 (최적화)
        overlay_bytes = io.BytesIO()
        overlay_img.save(overlay_bytes, format="PNG", optimize=True, compress_level=6)
        
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
    return {"status": "ok", "message": "Recycling Segmentation API v2.0 - DeepLabV3 + MobileViT 최적화"}

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

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "gpu_available": torch.cuda.is_available(),
        "classes_count": len(class_names)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
