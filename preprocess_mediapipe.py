from utils.mediapipeUtils import *
import os
import cv2

# 이미지 경로 설정
IMAGE_PATH = "dataset_50k/Validation/images/"
LABEL_PATH = "dataset_50k/Validation/labels/"
SAVE_IMAGE_PATH = "dataset_mediapipe/images/"
SAVE_LABEL_PATH = "dataset_mediapipe/labels/"

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(".jpg")]
label_files = [f for f in os.listdir(LABEL_PATH) if f.endswith(".txt")]

# 처리 시간 저장 리스트
processing_times = []
cnt = 0
total = len(image_files)

# 디렉토리 생성
os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
os.makedirs(SAVE_LABEL_PATH, exist_ok=True)

# 이미지 처리
for image_file in image_files:
    label_file = image_file.replace(".jpg", ".txt")
    cnt+=1
    image_path = os.path.join(IMAGE_PATH, image_file)
    frame = cv2.imread(image_path)
    label_path = os.path.join(LABEL_PATH, label_file)
    with open(label_path, 'r') as f:
        label = f.read()
    processor = FaceProcessor()
    processing_time = processor.preprocess_image(
        frame, 
        label,
        image_path=SAVE_IMAGE_PATH, 
        label_path=SAVE_LABEL_PATH, 
        cnt=cnt
    )
    if processing_time != -1:
        processing_times.append(processing_time)
    else:
        print(f"얼굴을 찾을 수 없습니다. {cnt}/{total}")
        continue
    # 처리 현황 출력 "진행 중(현재 진행 수 / 전체 수), 처리 시간: xxms"
    print(f"진행 중({cnt}/{total}), 처리 시간: {processing_time}ms")

# 처리 시간 평균 계산
processing_time = sum(processing_times) / len(processing_times)
print()
print(f"총 처리 시간: {sum(processing_times)}ms")
print(f"평균 처리 시간: {processing_time}ms")
print(f"총 {total}개 이미지 중 {len(processing_times)}개 처리 완료")
print(f"처리 완료율: {len(processing_times) / total * 100}%")