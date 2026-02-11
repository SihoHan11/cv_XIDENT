import os
import random
import cv2
from pathlib import Path
from tqdm import tqdm

def balance_dataset(base_path, move_path, sample_ratio=0.2):
    img_dir = Path(base_path) / 'images'
    lbl_dir = Path(base_path) / 'labels'
    move_img_dir = Path(move_path) / 'images'
    move_lbl_dir = Path(move_path) / 'labels'
    
    # 증강된 데이터를 저장할 경로
    aug_img_dir = img_dir + '/augmented'
    aug_lbl_dir = lbl_dir + '/augmented'

    label_files = list(lbl_dir.glob('*.txt'))
    
    minority_files = []
    majority_files = []

    # 저장된 클래스 파일 불러오기
    print("저장된 클래스 파일 불러오기...")
    with open(move_path / 'minority_classes.txt', 'r') as f:
        minority_files = [line.strip() for line in f.readlines()]
    
    with open(move_path / 'majority_classes.txt', 'r') as f:
        majority_files = [line.strip() for line in f.readlines()]
    
    print(f"소수 클래스 파일 수: {len(minority_files)}")
    print(f"다수 클래스 파일 수: {len(majority_files)}")

    if len(minority_files) == 0 or len(majority_files) == 0:
        print("클래스 분석 중...")
        for lbl_path in tqdm(label_files):
            with open(lbl_path, 'r') as f:
                classes = {line.split()[0] for line in f.readlines()}
            
            # 1: eye_closed, 2: mouth_opened 가 포함되어 있는가?
            if '1' in classes or '2' in classes:
                minority_files.append(lbl_path)
            else:
                majority_files.append(lbl_path)

        # 분석된 클래스 파일로 저장
        with open(move_path / 'minority_classes.txt', 'w') as f:
            for lbl_path in minority_files:
                f.write(f"{lbl_path}\n")
        
        with open(move_path / 'majority_classes.txt', 'w') as f:
            for lbl_path in majority_files:
                f.write(f"{lbl_path}\n")

    # 1. 다수 클래스 샘플링 (삭제 대신 별도 폴더 이동 권장)
    print(f"다수 클래스 샘플링 중... (유지 비율: {sample_ratio})")
    random.shuffle(majority_files)
    keep_count = int(len(majority_files) * sample_ratio)
    files_to_remove = majority_files[keep_count:]

    for lbl_path in tqdm(files_to_remove):
        img_path = img_dir / f"{lbl_path.stem}.jpg" # 확장자 확인 필요
        if img_path.exists():
            # os.remove(img_path) # 실제 삭제 시 사용
            # os.remove(lbl_path)
            # '/sampledData'로 이동
            os.rename(img_path, move_img_dir / f"{lbl_path.stem}.jpg")
            os.rename(lbl_path, move_lbl_dir / f"{lbl_path.stem}.txt")

    print(f"다수 클래스 샘플링 완료. (유지 비율: {sample_ratio})")


    # 2. 소수 클래스 증강 (Horizontal Flip)
    print("소수 클래스 증강 중 (Horizontal Flip)...")
    for lbl_path in tqdm(minority_files):
        img_path = img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists(): continue

        # 이미지 읽기 및 반전
        img = cv2.imread(str(img_path))
        flipped_img = cv2.flip(img, 1)
        
        # 반전된 이미지 저장
        new_stem = f"{lbl_path.stem}_flip"
        cv2.imwrite(str(aug_img_dir / f"{new_stem}.jpg"), flipped_img)

        # 라벨 반전 계산
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        new_annos = []
        for line in lines:
            cls, x, y, w, h = map(float, line.split())
            # YOLO x_center 반전: new_x = 1.0 - x
            new_x = 1.0 - x
            new_annos.append(f"{int(cls)} {new_x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
        with open(aug_lbl_dir / f"{new_stem}.txt", 'w') as f:
            f.write('\n'.join(new_annos))

    print("소수 클래스 증강 완료.")

# 밸런싱된 데이터 복구
def restore_dataset(base_path, move_path):
    img_dir = Path(base_path) / 'images'
    lbl_dir = Path(base_path) / 'labels'
    move_img_dir = Path(move_path) / 'images'
    move_lbl_dir = Path(move_path) / 'labels'

    # '/sampledData'로 이동된 이미지와 라벨 복구
    for img_path in move_img_dir.glob('*.jpg'):
        lbl_path = move_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            os.rename(img_path, img_dir / f"{img_path.stem}.jpg")
            os.rename(lbl_path, lbl_dir / f"{lbl_path.stem}.txt")
    
    # 증강된 이미지와 라벨 복구
    for img_path in img_dir.glob('*.jpg'):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            os.rename(img_path, img_dir / f"{img_path.stem}.jpg")
            os.rename(lbl_path, lbl_dir / f"{lbl_path.stem}.txt")

if __name__ == "__main__":
    # Training 폴더에 먼저 적용해봐
    balance_dataset('dataset/Training', 'dataset/sampledData/Training', sample_ratio=0.2)