import os
import random
import cv2
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

# 클래스 분석 유닛 (Worker)
def analyze_worker(lbl_path_str):
    try:
        with open(lbl_path_str, 'r') as f:
            classes = {line.split()[0] for line in f.readlines()}
        # 1: eye_closed, 2: mouth_opened
        return lbl_path_str, ('1' in classes or '2' in classes)
    except Exception:
        return lbl_path_str, False

# 파일 이동 유닛 (Worker)
def move_worker(task_info):
    lbl_path_str, img_dir_str, move_img_dir_str, move_lbl_dir_str = task_info
    lbl_path = Path(lbl_path_str)
    img_path = Path(img_dir_str) / f"{lbl_path.stem}.jpg"
    try:
        if img_path.exists():
            os.rename(img_path, Path(move_img_dir_str) / img_path.name)
            os.rename(lbl_path, Path(move_lbl_dir_str) / lbl_path.name)
            return True
    except Exception:
        pass
    return False

# 이미지 반전 및 라벨 생성 유닛 (Worker)
def augment_worker(task_info):
    lbl_path_str, img_dir_str, aug_img_dir_str, aug_lbl_dir_str = task_info
    lbl_path = Path(lbl_path_str)
    img_path = Path(img_dir_str) / f"{lbl_path.stem}.jpg"
    
    if not img_path.exists():
        return False

    try:
        img = cv2.imread(str(img_path))
        if img is None: return False
        flipped_img = cv2.flip(img, 1)
        
        new_stem = f"{lbl_path.stem}_flip"
        cv2.imwrite(str(Path(aug_img_dir_str) / f"{new_stem}.jpg"), flipped_img)

        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        new_annos = []
        for line in lines:
            parts = line.split()
            cls = parts[0]
            x, y, w, h = map(float, parts[1:])
            new_x = 1.0 - x
            new_annos.append(f"{cls} {new_x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            
        with open(Path(aug_lbl_dir_str) / f"{new_stem}.txt", 'w') as f:
            f.write('\n'.join(new_annos))
        return True
    except Exception:
        return False

def balance_dataset_pro(base_path, move_path, sample_ratio=0.2):
    base = Path(base_path)
    move = Path(move_path)
    
    img_dir, lbl_dir = base / 'images', base / 'labels'
    move_img_dir, move_lbl_dir = move / 'images', move / 'labels'
    aug_img_dir, aug_lbl_dir = img_dir / 'augmented', lbl_dir / 'augmented'

    for d in [move_img_dir, move_lbl_dir, aug_img_dir, aug_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    label_files = [str(p) for p in lbl_dir.glob('*.txt')]
    minority_files, majority_files = [], []

    print("[1/3] 클래스 병렬 분석 및 분류 중...")
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(analyze_worker, label_files, chunksize=100), total=len(label_files), desc="분석 중"))
    
    for path_str, is_minority in results:
        if is_minority:
            minority_files.append(path_str)
        else:
            majority_files.append(path_str)

    print(f"[2/3] 다수 클래스 병렬 샘플링 (유지 비율: {sample_ratio})...")
    random.shuffle(majority_files)
    keep_count = int(len(majority_files) * sample_ratio)
    to_move = majority_files[keep_count:]
    
    move_tasks = [(p, str(img_dir), str(move_img_dir), str(move_lbl_dir)) for p in to_move]
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(move_worker, move_tasks, chunksize=100), total=len(move_tasks), desc="이동 중"))

    print("[3/3] 소수 클래스 병렬 증강 시작 (Flip)...")
    aug_tasks = [(p, str(img_dir), str(aug_img_dir), str(aug_lbl_dir)) for p in minority_files]
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(augment_worker, aug_tasks, chunksize=50), total=len(aug_tasks), desc="증강 중"))

if __name__ == "__main__":
    balance_dataset_pro('dataset/Validation', 'dataset/sampledData/Validation', sample_ratio=0.2)
