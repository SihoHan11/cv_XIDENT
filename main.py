import os
import io
import datetime
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from supabase import create_client, Client
import cv2
from typing import Optional
from utils.camPredictUtils import *
from utils.mediapipeUtils import *

# ==============================
# 기본 설정
# ==============================
MODEL_PATH = "best.onnx"
SAVE_DIR = "drowsy_data"

# 보안을 위해 실제 서비스 시에는 환경변수 사용을 권장합니다.
SUPABASE_URL = "https://fplqfuggropbbibvzmtl.supabase.co"
SUPABASE_KEY = "sb_publishable_QXps8MSGAZJbaMMdfjICiQ_CVMBcV1c"

# ==============================
# FastAPI 및 미들웨어 설정
# ==============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 외부 서비스 연결 및 모델 로드
# ==============================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
model = YOLO(MODEL_PATH)
print(f"[모델 클래스 목록] {model.names}")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==============================
# 핵심 유틸리티 함수
# ==============================

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    YOLO 모델 입력 규격에 맞게 전처리
    """
    # 1. 흑백 변환 (학습 모델이 그레이스케일 기반인 경우)
    if img.mode != 'L':
        img = img.convert('L')
    
    # 2. 리사이징 (640px 너비 기준 비율 유지)
    target_width = 640
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    
    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return img_resized

def save_to_supabase(status: int, img_path: str):
    """
    분석 결과를 Supabase DB에 기록
    """
    try:
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "image_path": img_path,
        }
        supabase.table("drowsy_history").insert(data).execute()
    except Exception as e:
        print(f"Supabase Save Error: {e}")

def analyze_image(img_processed: Image.Image) -> list[bool]:
    # PIL Image를 OpenCV에서 처리 가능한 NumPy 배열(BGR)로 변환
    frame = np.array(img_processed)

    # 감지 결과 변수 초기화 (기본값 False)
    isEyeClosed = False
    isYawn = False
    isHeadDrop = False
    
    # 1. 전처리: 사용자 시점 최적화를 위한 좌우 반전
    frame = cv2.flip(frame, 1)

    # 2. MediaPipe 처리: 고개 떨굼, 좌표 데이터, 크롭 이미지 추출
    mpProcessed = mediapipeProcess.process_frame(frame)
    
    # 얼굴 미감지 시 즉시 종료 및 False 리스트 반환
    if mpProcessed == None :
        return [False, False, False]
        
    # 결과 데이터 분할 할당
    isHeadDrop, bbox_coords, crop_coords, cropped_frame = mpProcessed
    
    # 3. YOLO 추론 준비: 모델 학습 환경에 맞춰 흑백 변환
    frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

    # 4. YOLO 모델 추론: 눈 및 입 상태 분석
    results = model.predict(
        source=frame_gray,       # 크롭된 흑백 얼굴 이미지
        verbose=False,           # 로그 출력 억제
        save=False,              # 이미지 저장 비활성
        imgsz=320,               # 입력 사이즈 최적화
        conf=0.25,               # 탐지 임계값
        classes=[0, 1, 2, 3],    # 분석 대상 클래스 정의
        device="intel:gpu"       # 하드웨어 가속 사용
        )
    
    # 5. 후처리: 중복 탐지 박스 제거
    filtered_data = filter_overlapping_parts(results)

    # 6. 상태 판별: 탐지된 클래스별 카운트 및 조건 확인
    closed_eye_count = 0 
    
    for detected in filtered_data :
        class_id = detected[5] # 클래스 ID 추출
        
        # 눈 감김(1) 카운트
        if class_id == 1 :
            closed_eye_count += 1
            
        # 입 벌림(2) 탐지 시 하품 여부 최종 검증
        elif class_id == 2 and isYawning(detected, bbox_coords):
            isYawn = True 

    # 양쪽 눈 감김 판정 (2개 이상)
    if closed_eye_count >= 2 :
        isEyeClosed = True
    
    # 7. 최종 리스트 반환: [하품, 눈감김, 고개떨굼]
    return [isYawn, isEyeClosed, isHeadDrop]

# ==============================
# API 엔드포인트
# ==============================

@app.post("/analyze_raw")
async def analyze_raw(
    y_plane: UploadFile = File(...),
    uv_plane: UploadFile = File(None),
    width: int = Form(...),
    height: int = Form(...),
    row_stride: int = Form(...), 
    format: str = Form("nv21")
):
    y_bytes = await y_plane.read()
    y_flat = np.frombuffer(y_bytes, dtype=np.uint8)

    required_size = row_stride * height
    if len(y_flat) < required_size:
        y_flat = np.pad(y_flat, (0, required_size - len(y_flat)), 'constant')

    y_array = y_flat[:required_size].reshape((height, row_stride))[:, :width]

    # 2. 이미지화 및 회전 수정
    # NV21의 Y평면은 그레이스케일과 같으므로 바로 변환
    img_original = Image.fromarray(y_array, mode='L')
        
    # [중요] 안드로이드 전면 카메라는 보통 90도 회전되어 전송됨
    # 이미지가 옆으로 누워있으면 YOLO가 인식하지 못하므로 90도 회전
    img_original = img_original.rotate(-90, expand=True)

    # 3. 분석 수행
    img_processed = preprocess_image(img_original)
    status = analyze_image(img_processed)

    # 4. 결과 저장
    img_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(SAVE_DIR, img_filename)
    img_original.save(img_path) 

    save_to_supabase(status, img_path)

    return {"status": status}


@app.post("/analyze_jpeg")
async def analyze_jpeg(file: UploadFile = File(...)):
    """웹/일반 이미지 업로드 처리"""
    image_bytes = await file.read()
    img_original = Image.open(io.BytesIO(image_bytes))
    
    img_processed = preprocess_image(img_original)
    status = analyze_image(img_processed)
    
    img_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = os.path.join(SAVE_DIR, img_filename)
    img_original.save(img_path)
    
    save_to_supabase(status, img_path)
    return {"status": status}

@app.get("/")
async def health_check():
    return {"status": "ok", "endpoints": ["/analyze_raw", "/analyze_jpeg"]}


'''
def analyze_image(img_processed: Image.Image) -> int:
    YOLO 추론 및 상태 판별 (status 2: 졸음, 3: 정상)
    results = model.predict(source=img_processed, imgsz=640, conf=0.3)
    status = 3  # 기본값: 정상

    if len(results) > 0:
        boxes = results[0].boxes

        print(f"[DEBUG] 감지된 박스 수: {len(boxes)}")
        for box in boxes:
            cls_idx = int(box.cls.item())
            cls_name = model.names[cls_idx]
            conf = box.conf.item()
            print(f"[DEBUG] 클래스: {cls_name}, 신뢰도: {conf:.3f}")

            # 감지된 클래스가 eye_closed일 경우 졸음으로 판단
            if cls_name == "eye_closed" and conf >= 0.3:
                status = 2
                break
    else:
        print("[DEBUG] 감지된 박스 없음")
    return status
    '''