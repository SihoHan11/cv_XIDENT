# mediapipe + yolo26n hybrid model validation
# 서로의 단점을 보완하는 모델임을 증명해야함
import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from utils.mediapipeUtils import FaceProcessor

# 경로 설정
ORIG_IMAGE_DIR = "dataset_50k/Validation/images"
ORIG_LABEL_DIR = "dataset_50k/Validation/labels"

YOLO_SAVE_DIR_IMAGE = "dataset_yolo/failed/images"
YOLO_SAVE_DIR_LABEL = "dataset_yolo/failed/labels"
YOLO_PREPROCESSED_IMAGE = "dataset_yolo/failed/preprocessed/images"
YOLO_PREPROCESSED_LABEL = "dataset_yolo/failed/preprocessed/labels"

YOLO_320_PATH = 'runs/detect/train_100k/weights/320px_openvino_model'
YOLO_640_PATH = 'runs/detect/train_100k/weights/640px_openvino_model'

# 1. YOLO로 탐지 후 실패 사례(얼굴 미검출) 저장하는 함수
def yolo_failed_save(conf_threshold=0.25):
    print("Step 1: YOLO 탐지 실패 이미지 추출 중...")
    model = YOLO(YOLO_320_PATH)
    
    os.makedirs(YOLO_SAVE_DIR_IMAGE, exist_ok=True)
    os.makedirs(YOLO_SAVE_DIR_LABEL, exist_ok=True)

    image_files = [f for f in os.listdir(ORIG_IMAGE_DIR) if f.endswith(".jpg")]
    failed_count = 0

    for img_file in tqdm(image_files, desc="YOLO Detecting"):
        img_path = os.path.join(ORIG_IMAGE_DIR, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(ORIG_LABEL_DIR, label_file)

        if not os.path.exists(label_path):
            continue

        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # 탐지된 박스가 없는 경우 실패로 간주
        if len(results[0].boxes) == 0:
            shutil.copy(img_path, os.path.join(YOLO_SAVE_DIR_IMAGE, img_file))
            shutil.copy(label_path, os.path.join(YOLO_SAVE_DIR_LABEL, label_file))
            failed_count += 1

    print(f"추출 완료: {failed_count}개의 이미지가 YOLO 탐지에 실패했습니다.")
    return failed_count

# 2. YOLO가 탐지 실패한 이미지들을 정렬하여 MediaPipe로 전처리
def preprocess_yolo_failed_with_mp():
    print("\nStep 2: MediaPipe를 활용하여 실패 이미지 얼굴 정렬 중...")
    processor = FaceProcessor()
    
    os.makedirs(YOLO_PREPROCESSED_IMAGE, exist_ok=True)
    os.makedirs(YOLO_PREPROCESSED_LABEL, exist_ok=True)

    image_files = [f for f in os.listdir(YOLO_SAVE_DIR_IMAGE) if f.endswith(".jpg")]
    success_count = 0

    for i, img_file in enumerate(tqdm(image_files, desc="MediaPipe Aligning")):
        img_path = os.path.join(YOLO_SAVE_DIR_IMAGE, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(YOLO_SAVE_DIR_LABEL, label_file)

        frame = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            label_content = f.read()

        # FaceProcessor의 preprocess_image 사용 (정렬 및 저장 포함)
        # image_path/label_path 인자는 폴더 경로를 의미함 (mediapipeUtils.py 구현 기준)
        processing_time = processor.preprocess_image(
            frame, 
            label_content,
            image_path=YOLO_PREPROCESSED_IMAGE + "/", 
            label_path=YOLO_PREPROCESSED_LABEL + "/", 
            cnt=i+1
        )

        if processing_time != -1:
            success_count += 1
            
    print(f"전처리 완료: {success_count}개의 이미지가 성공적으로 정렬되었습니다.")
    return success_count

# 3. 정렬된 이미지들만 모아서 YOLO로 다시 성능 검증
def evaluate_hybrid_step2():
    print("\nStep 3: 정렬된 이미지 대상 YOLO 재검증 중...")
    '''
    # 임시 데이터 YAML 생성 (정렬된 데이터셋 전용)
    tmp_yaml = "data_yolo_failed_preprocessed.yaml"
    content = f"""
        path: {os.path.abspath('dataset_yolo/failed/preprocessed')}
        train: images
        val: images
        nc: 5
        names:
        0: 'eye_opened'
        1: 'eye_closed'
        2: 'mouth_opened'
        3: 'mouth_closed'
        4: 'face'
        """
        
    with open(tmp_yaml, 'w') as f:
        f.write(content)
    '''
    model = YOLO(YOLO_320_PATH)
    metrics = model.val(
        data="data_yolo_failed_preprocessed.yaml",
        imgsz=320,
        save=True,
        project='runs/hybrid_val',
        name='yolo_failed_aligned',
        device='intel:gpu' # 사용 가능한 디바이스로 설정
    )
    
    print("\n" + "="*40)
    print("하이브리드 검증 결과")
    print("="*40)
    print(f"mAP50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")
    print("="*40)

# mediapipe가 탐지 실패한 이미지들만 모아서 yolo로 탐지 (기존 기능 유지)
def mediapipe_failed_yolo_detect():
    print("\nMediaPipe 실패 이미지 대상 YOLO 검증...")
    model = YOLO(YOLO_640_PATH)
    metrics = model.val(
        data="data_hybrid.yaml",
        imgsz=640,
        save=True,
        project='dataset_mediapipe/failed',
        name='yolo_on_mp_failed',
        device='intel:gpu'
    )
    print(metrics)
'''
# main
if __name__ == "__main__":
    # 1. YOLO 실패 이미지 추출
    failed_count = yolo_failed_save()
    
    if failed_count > 0:
        # 2. 실패 이미지에 대해 MediaPipe 정렬 적용
        aligned_count = preprocess_yolo_failed_with_mp()
        
        if aligned_count > 0:
            # 3. 정렬된 이미지로 다시 성능 검증
            evaluate_hybrid_step2()
        else:
            print("MediaPipe가 정렬에 성공한 이미지가 없습니다.")
    else:
        print("YOLO 탐지에 실패한 이미지가 없어 하이브리드 검증을 종료합니다.")'''

if __name__ == "__main__":
    evaluate_hybrid_step2()