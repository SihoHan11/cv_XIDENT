from utils.mediapipeUtils import *
import os
import cv2

# 이미지 경로 설정
IMAGE_PATH = "dataset_mediapipe/images/"
LABEL_PATH = "dataset_mediapipe/labels/"
SAVE_PATH = "dataset_mediapipe/visualize/"

# 처리된 이미지와 라벨이 일치하는지 시각화 후 저장하는 코드
# 50개의 이미지만 테스트

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith((".jpg", ".png", ".jpeg"))]
os.makedirs(SAVE_PATH, exist_ok=True)

cnt = 0
total = 50

print(f"--- 시각화 검증 시작 (목표: {total}개) ---")

for img_name in image_files:
    if cnt >= total:
        break
    
    # 대응하는 라벨 파일 경로 확인
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_file_path = os.path.join(LABEL_PATH, label_name)
    
    if not os.path.exists(label_file_path):
        continue

    # 이미지 읽기
    img_path = os.path.join(IMAGE_PATH, img_name)
    image = cv2.imread(img_path)
    if image is None: continue
    
    h, w, _ = image.shape

    # 라벨 파일 읽기 및 그리기
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # YOLO format: class_id, x_center, y_center, width, height (normalized)
            class_id = parts[0]
            x_c, y_c, bw, bh = map(float, parts[1:5])
            
            # 픽셀 좌표로 변환
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            
            # 바운딩 박스 시각화 (선명한 네온 그린 컬러)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"ID: {class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과물 저장
    save_file_path = os.path.join(SAVE_PATH, f"vis_{img_name}")
    cv2.imwrite(save_file_path, image)
    
    cnt += 1
    print(f"[{cnt}/{total}] 저장 완료: {save_file_path}")

print(f"\n검증 작업이 끝났어. 결과는 '{SAVE_PATH}'에서 직접 확인해 봐.")