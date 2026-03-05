from utils.mediapipeUtils import FaceProcessor
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 이미지 경로 설정
IMAGE_PATH = "dataset_50k/Validation/images/"
LABEL_PATH = "dataset_50k/Validation/labels/"
SAVE_IMAGE_PATH = "dataset_mediapipe/Validation/images/"
SAVE_LABEL_PATH = "dataset_mediapipe/Validation/labels/"
SAVE_FAILED_IMAGE_PATH = "dataset_mediapipe/failed/images/"
SAVE_FAILED_LABEL_PATH = "dataset_mediapipe/failed/labels/"

# 전역 프로세서 변수 (워커 프로세스에서 재사용)
processor = None

def init_worker():
    """워커 프로세스 초기화: FaceProcessor를 한 번만 생성합니다."""
    global processor
    processor = FaceProcessor()

def process_image_task(args):
    """단일 이미지 처리 태스크"""
    img_file, cnt = args
    label_file = img_file.replace(".jpg", ".txt")
    image_path = os.path.join(IMAGE_PATH, img_file)
    label_orig_path = os.path.join(LABEL_PATH, label_file)
    
    frame = cv2.imread(image_path)
    if frame is None:
        return -1, img_file
        
    if not os.path.exists(label_orig_path):
        return -1, img_file

    with open(label_orig_path, 'r') as f:
        label_content = f.read()

    # init_worker에서 생성된 전역 processor 사용
    processing_time = processor.preprocess_image(
        frame, 
        label_content,
        image_path=SAVE_IMAGE_PATH, 
        label_path=SAVE_LABEL_PATH, 
        cnt=cnt
    )
    
    if processing_time == -1:
        # 실패 시 실패 폴더에 저장
        os.makedirs(SAVE_FAILED_IMAGE_PATH, exist_ok=True)
        os.makedirs(SAVE_FAILED_LABEL_PATH, exist_ok=True)
        cv2.imwrite(os.path.join(SAVE_FAILED_IMAGE_PATH, img_file), frame)
        with open(os.path.join(SAVE_FAILED_LABEL_PATH, label_file), 'w') as f:
            f.write(label_content)
        return -1, img_file
        
    return processing_time, img_file

def main():
    # 디렉토리 생성
    os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
    os.makedirs(SAVE_LABEL_PATH, exist_ok=True)

    # 이미지 파일 목록 가져오기
    if not os.path.exists(IMAGE_PATH):
        print(f"경로를 찾을 수 없습니다: {IMAGE_PATH}")
        return

    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(".jpg")]
    total = len(image_files)
    
    if total == 0:
        print("처리할 이미지 파일이 없습니다.")
        return

    print(f"총 {total}개의 이미지 처리를 시작합니다. (병렬 처리)")

    processing_times = []
    failed_count = 0
    
    # 병렬 처리 실행
    # max_workers는 시스템 CPU 코어 수에 따라 자동 조절됨
    with ProcessPoolExecutor(initializer=init_worker) as executor:
        # 작업 인자 리스트 생성 (파일명, 고유 번호)
        tasks = [(img_file, i + 1) for i, img_file in enumerate(image_files)]
        
        # tqdm으로 진행률 표시하며 결과 수집
        # map은 순서를 보장하지만, 작업은 병렬로 진행됨
        for processing_time, img_file in tqdm(executor.map(process_image_task, tasks), total=total, desc="Processing"):
            if processing_time != -1:
                processing_times.append(processing_time)
            else:
                failed_count += 1

    # 결과 요약 출력
    print("\n" + "="*30)
    print("처리 결과 요약")
    print("="*30)
    if processing_times:
        total_time = sum(processing_times)
        avg_time = total_time / len(processing_times)
        print(f"총 누적 처리 시간: {total_time:.2f}ms")
        print(f"평균 처리 시간: {avg_time:.2f}ms")
    
    print(f"전체 파일 수: {total}")
    print(f"성공: {len(processing_times)}개")
    print(f"실패: {failed_count}개")
    if total > 0:
        print(f"최종 완료율: {len(processing_times) / total * 100:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()
