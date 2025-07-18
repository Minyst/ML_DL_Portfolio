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
    title="Recycling Segmentation API - Final Version",
    description="재활용품 분류를 위한 Semantic Segmentation API (완전판)",
    version="2.0.0"
)

# CORS 설정 (모바일 앱 완전 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (모바일 앱용)
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# 모델 및 전역 변수
MODEL_PATH = "./best_model"
model = None
processor = None

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 모델 로드 및 최적화"""
    global model, processor
    try:
        logger.info("🚀 모델 로드 시작...")
        model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH)
        processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        model.eval()
        
        # GPU/CPU 최적화
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ GPU 모드로 모델 로드 완료!")
        else:
            # CPU 최적화 설정
            torch.set_num_threads(4)
            torch.backends.mkldnn.enabled = True
            
            try:
                # JIT 컴파일 시도 (속도 향상)
                model = torch.jit.script(model)
                logger.info("✅ CPU 모드로 모델 로드 완료! (JIT 최적화)")
            except Exception as jit_error:
                logger.warning(f"JIT 컴파일 실패, 일반 모드로 진행: {jit_error}")
                logger.info("✅ CPU 모드로 모델 로드 완료! (일반 모드)")
                
        logger.info(f"🔧 모델 설정 완료 - GPU: {torch.cuda.is_available()}")
        
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {str(e)}")

# 클래스 정의
class_names = [
    "background", "can", "glass", "paper", "plastic", "styrofoam", "vinyl"
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

def load_font():
    """안전한 폰트 로드"""
    try:
        return ImageFont.truetype("arial.ttf", 18)
    except:
        try:
            return ImageFont.truetype("Arial.ttf", 18)
        except:
            try:
                return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            except:
                return ImageFont.load_default()

def smooth_mask(mask, sigma=0.8):
    """마스크 부드럽게 처리"""
    mask = np.array(mask) if hasattr(mask, 'size') else mask
    smoothed = np.zeros_like(mask, dtype=np.float32)
    
    for class_id in range(len(class_names)):
        class_mask = (mask == class_id).astype(np.float32)
        if class_mask.sum() > 0:
            smoothed_class = gaussian_filter(class_mask, sigma=sigma)
            smoothed[smoothed_class > 0.5] = class_id
    
    return smoothed.astype(np.uint8)

def add_contours(image, mask):
    """고품질 테두리 추가"""
    result = np.array(image)
    
    for class_id in range(1, len(class_names)):
        if (mask == class_id).sum() == 0:
            continue
            
        # 클래스별 마스크 생성
        class_mask = (mask == class_id).astype(np.uint8) * 255
        
        # 컨투어 찾기
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            color = class_colors_bright[class_id]
            # 외곽선 그리기 (굵게)
            cv2.drawContours(result, contours, -1, color, 3)
            # 내부선 그리기 (얇게)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
    
    return Image.fromarray(result)

def create_visualization(image, mask):
    """고품질 시각화 생성 (촬영용)"""
    # 마스크 부드럽게 처리
    mask_smooth = smooth_mask(mask, sigma=0.8)
    
    overlay = image.copy().convert("RGB")
    pred = Image.new("RGB", image.size, (0, 0, 0))

    draw_overlay = ImageDraw.Draw(overlay)
    draw_pred = ImageDraw.Draw(pred)
    font = load_font()

    mask_smooth = np.array(mask_smooth)
    min_area = 300  # 최소 면적 (노이즈 제거)

    for class_id in range(len(class_names)):
        class_mask = (mask_smooth == class_id)
        area = class_mask.sum()
        
        if area < min_area and class_id > 0:  # background는 면적 제한 없음
            continue

        color = class_colors_bright[class_id]

        # Overlay 생성 (현실 배경 + 색상 오버레이)
        if class_id > 0:  # background 제외
            overlay_arr = np.array(overlay)
            alpha = 0.35  # 적당한 투명도
            for c in range(3):
                overlay_arr[:, :, c][class_mask] = (
                    overlay_arr[:, :, c][class_mask] * (1 - alpha) + 
                    color[c] * alpha
                )
            overlay = Image.fromarray(overlay_arr)

        # Prediction 생성 (검은 배경 + 색상 채우기)
        pred_arr = np.array(pred)
        for c in range(3):
            pred_arr[:, :, c][class_mask] = color[c]
        pred = Image.fromarray(pred_arr)

        # 라벨 추가 (background 제외)
        if class_id > 0 and area >= min_area:
            yx = np.argwhere(class_mask)
            if len(yx) > 0:
                y_mean, x_mean = yx.mean(axis=0).astype(int)
                label = class_names[class_id]

                # 라벨 배경 및 텍스트 그리기
                for draw in [draw_pred, draw_overlay]:
                    draw.rectangle([(x_mean - 45, y_mean - 18), (x_mean + 45, y_mean + 18)], fill="black")
                    draw.text((x_mean - 35, y_mean - 15), label, fill="white", font=font)

    # 테두리 추가 (prediction에만)
    pred = add_contours(pred, mask_smooth)
    
    return pred, overlay

def create_overlay_only(image, mask):
    """실시간용 overlay 생성 (속도 최적화)"""
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = load_font()
    
    mask = np.array(mask)
    min_area = 200  # 실시간용 최소 면적
    
    for class_id in range(1, len(class_names)):  # background 제외
        class_mask = (mask == class_id)
        area = class_mask.sum()
        
        if area < min_area:  # 작은 영역 무시
            continue
            
        color = class_colors_bright[class_id]
        
        # 색상 오버레이
        overlay_arr = np.array(overlay)
        alpha = 0.3
        for c in range(3):
            overlay_arr[:, :, c][class_mask] = (
                overlay_arr[:, :, c][class_mask] * (1 - alpha) + 
                color[c] * alpha
            )
        overlay = Image.fromarray(overlay_arr)

        # 중심 라벨만 표시
        yx = np.argwhere(class_mask)
        if len(yx) > 0:
            y_mean, x_mean = yx.mean(axis=0).astype(int)
            label = class_names[class_id]
            
            # 라벨 배경 및 텍스트
            draw.rectangle([(x_mean - 40, y_mean - 15), (x_mean + 40, y_mean + 15)], fill="black")
            draw.text((x_mean - 30, y_mean - 12), label, fill="white", font=font)
    
    return overlay

def predict_image(image_bytes, realtime=False):
    """이미지 예측 함수 (최적화된 버전)"""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")
    
    try:
        # 이미지 로드 및 전처리
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_size = image.size
        
        logger.info(f"📷 입력 이미지 크기: {original_size}")
        
        # 학습 시와 동일한 크기로 처리 (512x512)
        target_size = 512
        image_for_processing = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        logger.info(f"🔄 처리 이미지 크기: {image_for_processing.size}")
        
        # 모델 입력 준비
        inputs = processor(images=image_for_processing, return_tensors="pt")

        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # 모델 추론
        with torch.no_grad():
            outputs = model(**inputs)
            
            # 소프트맥스 적용 (부드러운 경계)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1).squeeze().cpu().numpy()

        logger.info(f"🧠 모델 출력 크기: {preds.shape}")

        # 원본 크기로 리사이징
        preds = preds.astype(np.uint8)
        preds_resized = cv2.resize(preds, original_size, interpolation=cv2.INTER_CUBIC)
        preds_resized = Image.fromarray(preds_resized)
        
        logger.info(f"✅ 최종 출력 크기: {preds_resized.size}")
        
        if realtime:
            # 실시간 모드: 단순 overlay
            overlay = create_simple_overlay(image, preds_resized)
            return None, overlay
        else:
            # 일반 모드: 단순 둘 다
            pred, overlay = create_simple_visualization(image, preds_resized)
            return pred, overlay
            
    except Exception as e:
        logger.error(f"❌ 예측 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """일반 예측 API (촬영용 - 고품질)"""
    # 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    if file.size and file.size > 15 * 1024 * 1024:  # 15MB 제한
        raise HTTPException(status_code=413, detail="파일 크기가 15MB를 초과합니다")
    
    try:
        start_time = time.time()
        image_bytes = await file.read()
        logger.info(f"📁 업로드된 파일 크기: {len(image_bytes)} bytes")
        
        pred_img, overlay_img = predict_image(image_bytes, realtime=False)

        # base64 인코딩
        pred_bytes = io.BytesIO()
        overlay_bytes = io.BytesIO()
        pred_img.save(pred_bytes, format="PNG", optimize=True)
        overlay_img.save(overlay_bytes, format="PNG", optimize=True)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️ 일반 처리 시간: {processing_time:.3f}초")

        return {
            "success": True,
            "prediction": base64.b64encode(pred_bytes.getvalue()).decode("utf-8"),
            "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
            "classes": class_names,
            "processing_time": round(processing_time, 3)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 일반 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@app.post("/predict-realtime")
async def predict_realtime(file: UploadFile = File(...)):
    """실시간 예측 API (속도 최적화)"""
    # 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")
    
    if file.size and file.size > 10 * 1024 * 1024:  # 10MB 제한
        raise HTTPException(status_code=413, detail="파일 크기가 10MB를 초과합니다")
    
    try:
        start_time = time.time()
        image_bytes = await file.read()
        
        _, overlay_img = predict_image(image_bytes, realtime=True)

        # base64 인코딩 (압축 최적화)
        overlay_bytes = io.BytesIO()
        overlay_img.save(overlay_bytes, format="PNG", optimize=True, compress_level=6)
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ 실시간 처리 시간: {processing_time:.3f}초")

        return {
            "success": True,
            "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
            "processing_time": round(processing_time, 3)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 실시간 예측 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@app.get("/")
async def root():
    """메인 헬스 체크"""
    return {
        "status": "🚀 Recycling Segmentation API v2.0 - RUNNING",
        "message": "DeepLabV3 + MobileViT 최적화 완료",
        "endpoints": {
            "predict": "POST /predict (촬영용 고품질)",
            "predict-realtime": "POST /predict-realtime (실시간 최적화)",
            "health": "GET /health (상세 상태)",
            "classes": "GET /classes (클래스 정보)"
        }
    }

@app.head("/")
async def root_head():
    """HEAD 요청 지원"""
    return {}

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "classes_count": len(class_names),
        "classes": class_names,
        "system_info": {
            "torch_version": torch.__version__,
            "mkldnn_enabled": torch.backends.mkldnn.enabled if hasattr(torch.backends, 'mkldnn') else False,
            "num_threads": torch.get_num_threads()
        }
    }

@app.get("/classes")
async def get_classes():
    """클래스 정보 조회"""
    return {
        "classes": class_names,
        "colors": class_colors_bright,
        "count": len(class_names)
    }

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """CORS Preflight 요청 처리"""
    return {}

# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"🚨 글로벌 에러: {exc}")
    return {
        "success": False,
        "error": str(exc),
        "detail": "서버 내부 오류가 발생했습니다."
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🌟 서버 시작 - Port: {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
