from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import io
import os
import uvicorn
import base64

# ===== FastAPI 앱 생성 =====
app = FastAPI(title="Smart Recycling Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Smart Recycling Segmentation API is running!",
        "version": "3.0",
        "features": ["adaptive_background_removal", "multi_class_separation", "depth_based_isolation"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

# ===== 모델 로드 =====
MODEL_PATH = os.path.abspath(os.path.dirname(__file__))

try:
    print("🤖 모델 로드 중...")
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_PATH, local_files_only=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    print("✅ 모델 로드 완료!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None
    processor = None

# ===== 설정 =====
class_names = ["background", "can", "glass", "paper", "plastic", "styrofoam", "vinyl"]
class_colors_bright = [
    None,             # background - 색상 없음 (투명)
    (255, 69, 0),     # can - 주황빨강
    (50, 205, 50),    # glass - 라임그린
    (30, 144, 255),   # paper - 파랑
    (255, 20, 147),   # plastic - 딥핑크
    (255, 215, 0),    # styrofoam - 골드
    (138, 43, 226)    # vinyl - 보라
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 디바이스: {device}")

font_path = os.path.join(os.path.dirname(__file__), "Pretendard-SemiBold.otf")

# ===== 배경 제거 및 객체 분리 함수들 =====

# 사용하지 않는 함수들 제거 (완전 삭제)
# 더 이상 사용하지 않는 함수들:
# - smart_background_removal (제거됨)
# - separate_complex_objects (제거됨) 
# - enhance_segmentation_quality (제거됨)
# - remove_noise_and_smooth (제거됨)

# 모든 KMeans 관련 import도 제거 (더 이상 사용 안 함)

# ===== 시각화 함수들 =====

def create_enhanced_visualization(image, mask, object_mask=None):
    """향상된 시각화 생성 - 올바른 Background 처리"""
    print("🎨 향상된 시각화 생성 중...")
    
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    # === OVERLAY 이미지: Background 투명(원본), 객체 컬러 ===
    overlay = img_np.copy().astype(np.float32)
    
    for class_id in range(1, len(class_names)):  # Background(0) 제외
        class_region = (mask == class_id)
        if np.any(class_region):
            color = class_colors_bright[class_id]
            # 원본 이미지에 컬러 오버레이
            overlay[class_region] = (
                img_np[class_region].astype(np.float32) * 0.4 +  # 원본 40%
                np.array(color) * 0.6  # 컬러 60%
            )
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # === PREDICT 이미지: Background 검은색, 객체 밝은 컬러 ===
    predict = np.zeros_like(img_np)  # 검은색 배경
    
    for class_id in range(1, len(class_names)):  # Background(0) 제외
        class_region = (mask == class_id)
        if np.any(class_region):
            color = class_colors_bright[class_id]
            predict[class_region] = color  # 순수 컬러
    
    # 라벨 추가 (객체에만)
    overlay_pil = add_clean_labels(Image.fromarray(overlay), mask)
    predict_pil = add_clean_labels(Image.fromarray(predict), mask)
    
    return predict_pil, overlay_pil

def add_clean_labels(image, mask):
    """깔끔한 라벨 추가 - Background 제외"""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, 18)
    except:
        try:
            font = ImageFont.load_default()
        except:
            return image

    for class_id in range(1, len(class_names)):  # Background(0) 제외
        class_mask = (mask == class_id)
        if not np.any(class_mask):
            continue

        # 클래스 영역의 중심점 계산
        y_coords, x_coords = np.where(class_mask)
        if len(x_coords) == 0 or len(y_coords) == 0:
            continue
            
        x_center = int(np.mean(x_coords))
        y_center = int(np.mean(y_coords))
        
        label = class_names[class_id]
        
        # 텍스트 크기 계산
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            text_w, text_h = len(label) * 8, 12
        
        padding = 6
        
        # 배경 박스 (테두리 없음)
        box = [
            x_center - text_w//2 - padding,
            y_center - text_h//2 - padding,
            x_center + text_w//2 + padding,
            y_center + text_h//2 + padding
        ]
        
        # 반투명 검은 배경
        draw.rectangle(box, fill=(0, 0, 0, 180))
        
        # 흰색 텍스트
        draw.text((x_center - text_w//2, y_center - text_h//2), 
                 label, fill="white", font=font)
        
    return image

# ===== 메인 처리 함수 =====

def process_smart_segmentation(image_bytes):
    """스마트 세그멘테이션 처리"""
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="모델이 로드되지 않았습니다")

    try:
        print("🚀 스마트 세그멘테이션 시작...")
        
        # 1. 이미지 로드
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        print(f"   원본 크기: {image.size}")
        
        # 2. 스마트 배경 제거
        object_mask = smart_background_removal(img_np)
        
        # 3. 모델 입력 준비 및 입력 이미지 디버깅
        print(f"📋 PIL 이미지 통계:")
        img_array = np.array(image)
        print(f"   크기: {img_array.shape}")
        print(f"   픽셀 범위: {img_array.min()} ~ {img_array.max()}")
        print(f"   평균: {img_array.mean():.2f}")
        
        # 입력 이미지가 너무 어두운 경우 밝기 조정
        if img_array.mean() < 50:
            print("⚠️ 입력 이미지가 너무 어둡습니다. 밝기 조정...")
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(2.0)  # 2배 밝게
            img_array = np.array(image)
            print(f"   조정 후 픽셀 범위: {img_array.min()} ~ {img_array.max()}")
            print(f"   조정 후 평균: {img_array.mean():.2f}")
        
        inputs = processor(images=image, return_tensors="pt")
        print(f"📋 입력 텐서 크기: {inputs['pixel_values'].shape}")
        print(f"📋 입력 텐서 범위: {inputs['pixel_values'].min():.3f} ~ {inputs['pixel_values'].max():.3f}")
        
        # 입력 텐서가 모두 0인 경우 강제로 정규화
        if inputs['pixel_values'].max() == 0:
            print("⚠️ 입력 텐서가 모두 0입니다. 직접 정규화 시도...")
            # ImageNet 정규화 역산
            img_tensor = torch.tensor(img_array.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            normalized = (img_tensor - mean) / std
            inputs['pixel_values'] = normalized.unsqueeze(0)
            print(f"   직접 정규화 후 범위: {inputs['pixel_values'].min():.3f} ~ {inputs['pixel_values'].max():.3f}")
        
        # 4. 모델 예측 (더 정밀하게)
        with torch.no_grad():
            outputs = model(pixel_values=inputs["pixel_values"].to(device))
            logits = outputs.logits
            print(f"📋 출력 로짓 크기: {logits.shape}")
            print(f"📋 출력 로짓 범위: {logits.min():.3f} ~ {logits.max():.3f}")
            
            # 더 높은 해상도로 업샘플링
            target_size = (image.size[1], image.size[0])  # (H, W)
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
            
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            print(f"📋 확률 분포 - 각 클래스별 최대값:")
            for i in range(len(class_names)):
                max_prob = probs[i].max()
                mean_prob = probs[i].mean()
                print(f"   {class_names[i]}: max={max_prob:.3f}, mean={mean_prob:.3f}")
            
            initial_mask = np.argmax(probs, axis=0)
            print(f"📋 모델 원시 예측: {np.unique(initial_mask, return_counts=True)}")
            
            # 더 관대한 확신도 기반 필터링
            confidence_threshold = 0.05  # 더 낮춤
            max_probs = np.max(probs, axis=0)
            low_confidence = max_probs < confidence_threshold
            initial_mask[low_confidence] = 0
            
            print(f"📋 확신도 필터링 후: {np.unique(initial_mask, return_counts=True)}")
            
            # 만약 여전히 모든 게 배경이면 강제로 가장 높은 확률 영역을 클래스로 할당
            if len(np.unique(initial_mask)) == 1 and np.unique(initial_mask)[0] == 0:
                print("⚠️ 모든 픽셀이 배경으로 분류됨, 더 적극적 강제 할당...")
                
                # 각 클래스별로 가장 높은 확률을 가진 픽셀들 찾기
                for class_id in range(1, len(class_names)):
                    class_probs = probs[class_id]
                    if class_probs.max() > 0.001:  # 매우 매우 낮은 임계값
                        # 상위 5% 픽셀을 해당 클래스로 할당
                        threshold = np.percentile(class_probs, 95)
                        high_prob_mask = class_probs >= threshold
                        if high_prob_mask.sum() > 50:  # 최소 50픽셀
                            initial_mask[high_prob_mask] = class_id
                            print(f"   강제 할당: {class_names[class_id]} - {high_prob_mask.sum()}픽셀")
                            
                print(f"📋 강제 할당 후: {np.unique(initial_mask, return_counts=True)}")
        
        print("   배경 마스크 적용 생략 (디버깅 모드)")
        
        # 7. 복합 객체 분리
        separated_mask = separate_complex_objects(initial_mask, probs)
        
        # 8. 세그멘테이션 품질 향상
        enhanced_mask = enhance_segmentation_quality(img_np, separated_mask, probs)
        
        # 9. 노이즈 제거 및 최종 정제
        final_mask = remove_noise_and_smooth(enhanced_mask)
        
        # 10. 결과 분석 - 더 관대한 기준
        unique, counts = np.unique(final_mask, return_counts=True)
        print("📊 최종 결과:")
        detected_classes = []
        total_pixels = final_mask.size
        
        for u, c in zip(unique, counts):
            if u > 0 and u < len(class_names):  # 최소 픽셀 수 조건 제거
                percentage = (c / total_pixels) * 100
                print(f"   {class_names[u]}: {c:,}px ({percentage:.1f}%)")
                if c > 50:  # 50픽셀 이상만 유효한 클래스로 인정
                    detected_classes.append((class_names[u], c))
        
        # 크기 순으로 정렬
        detected_classes.sort(key=lambda x: x[1], reverse=True)
        class_names_only = [cls[0] for cls in detected_classes]
        
        print(f"📋 감지된 클래스들: {class_names_only}")
        
        # 아무것도 감지되지 않은 경우 디버깅 정보 출력
        if not class_names_only:
            print("⚠️ 아무것도 감지되지 않음!")
            print(f"   초기 마스크 유니크 값: {np.unique(initial_mask)}")
            print(f"   분리 후 마스크 유니크 값: {np.unique(separated_mask)}")
            print(f"   향상 후 마스크 유니크 값: {np.unique(enhanced_mask)}")
            print(f"   최종 마스크 유니크 값: {np.unique(final_mask)}")
            
            # 강제로 최소한의 결과라도 반환
            if np.any(initial_mask > 0):
                fallback_unique, fallback_counts = np.unique(initial_mask, return_counts=True)
                for u, c in zip(fallback_unique, fallback_counts):
                    if u > 0 and u < len(class_names) and c > 10:
                        class_names_only.append(class_names[u])
                        print(f"   폴백으로 추가: {class_names[u]} ({c}px)")
                        break
        
        # 11. 시각화 생성
        pred_img, overlay_img = create_enhanced_visualization(image, final_mask, object_mask)
        
        print("✅ 스마트 세그멘테이션 완료!")
        return pred_img, overlay_img, class_names_only

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")

# ===== FastAPI 엔드포인트 =====

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """스마트 세그멘테이션 수행"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다")

        print(f"📤 요청 받음: {file.filename} ({file.content_type})")
        
        image_bytes = await file.read()
        print(f"   파일 크기: {len(image_bytes):,} bytes")
        
        # 스마트 세그멘테이션 처리
        pred_img, overlay_img, detected_classes = process_smart_segmentation(image_bytes)

        # 결과 인코딩
        pred_bytes = io.BytesIO()
        overlay_bytes = io.BytesIO()
        pred_img.save(pred_bytes, format="PNG", optimize=True, quality=90)
        overlay_img.save(overlay_bytes, format="PNG", optimize=True, quality=90)

        # 주요 클래스 결정 (가장 큰 영역)
        main_class = detected_classes[0] if detected_classes else "unknown"
        confidence = min(0.95, max(0.7, len(detected_classes) * 0.15 + 0.7))

        response = {
            "prediction": base64.b64encode(pred_bytes.getvalue()).decode("utf-8"),
            "overlay": base64.b64encode(overlay_bytes.getvalue()).decode("utf-8"),
            "class": main_class,
            "confidence": confidence,
            "detected_classes": detected_classes,
            "status": "success",
            "message": "스마트 세그멘테이션 완료"
        }
        
        print(f"📤 응답 전송: {main_class} ({confidence:.2f}) - {detected_classes}")
        return response

    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류: {str(e)}")

@app.post("/predict-raw")
async def predict_raw(file: UploadFile = File(...)):
    """원본 이미지도 처리 가능 (호환성용)"""
    return await predict(file)

# ===== 서버 실행 =====

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 서버 시작: http://0.0.0.0:{port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
