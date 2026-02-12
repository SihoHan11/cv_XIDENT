import os
import random
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

# 1. 라벨 분석 워커 (병렬 처리용)
def analyze_worker(lbl_path_str):
    lbl_path = Path(lbl_path_str)
    try:
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            classes = {line.split()[0] for line in lines} if lines else set()
        
        # 가치 평가 점수 (높을수록 우선 추출)
        if '2' in classes or '3' in classes: score = 3  # Mouth 관련 (최우선)
        elif '1' in classes: score = 2                  # Eye Closed
        elif not classes: score = 0                     # Background
        else: score = 1                                 # Normal (Eye Opened/Face)
        
        return (lbl_path.stem, score)
    except:
        return (lbl_path.stem, -1)

# 2. 파일 복사 워커 (병렬 처리용)
def copy_worker(task):
    src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, stem = task
    try:
        shutil.copy(src_img_dir / f"{stem}.jpg", dst_img_dir / f"{stem}.jpg")
        shutil.copy(src_lbl_dir / f"{stem}.txt", dst_lbl_dir / f"{stem}.txt")
        return True
    except:
        return False

def ultra_fast_sampling(src_base, dst_base, target_total=50000, bg_ratio=0.1):
    src_base, dst_base = Path(src_base), Path(dst_base)
    src_img, src_lbl = src_base / 'images', src_base / 'labels'
    dst_img, dst_lbl = dst_base / 'images', dst_base / 'labels'

    for d in [dst_img, dst_lbl]: d.mkdir(parents=True, exist_ok=True)

    # --- 1단계: 병렬 라벨 분석 ---
    print(f"\n[1/3] {len(list(src_lbl.glob('*.txt')))}개 라벨 병렬 분석 시작...")
    lbl_paths = [str(p) for p in src_lbl.glob('*.txt')]
    
    analysis_results = []
    with ProcessPoolExecutor() as executor:
        analysis_results = list(tqdm(executor.map(analyze_worker, lbl_paths, chunksize=500), 
                                     total=len(lbl_paths), desc="라벨 분석 중"))

    # 결과 분류
    categories = {3: [], 2: [], 1: [], 0: []}
    for stem, score in analysis_results:
        if score in categories:
            categories[score].append(stem)

    # --- 2단계: 전략적 샘플 선별 ---
    print("[2/3] 정예 멤버 선별 중...")
    bg_target = int(target_total * bg_ratio)
    fg_target = target_total - bg_target

    selected_stems = []
    
    # 포어그라운드 (Mouth -> Eye Closed -> Normal 순으로 채움)
    priority_fg = categories[3] + categories[2] + categories[1]
    random.shuffle(priority_fg)
    selected_stems.extend(priority_fg[:fg_target])

    # 백그라운드 샘플 추가
    random.shuffle(categories[0])
    selected_stems.extend(categories[0][:bg_target])

    # 선택된 샘플 클래스 별 개수 출력
    print("\n[Selected Samples Class Distribution]")
    for cls_id, cls_name in enumerate(['eye_opened', 'eye_closed', 'mouth_opened', 'mouth_closed', 'face']):
        count = sum(1 for stem in selected_stems if str(cls_id) in open(src_lbl / f"{stem}.txt").read())
        print(f"{cls_name}: {count}")

    # --- 3단계: 병렬 파일 복사 ---
    print(f"[3/3] {len(selected_stems)}개 파일 병렬 복사 중...")
    copy_tasks = [(src_img, src_lbl, dst_img, dst_lbl, s) for s in selected_stems]
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(copy_worker, copy_tasks, chunksize=100), 
                  total=len(copy_tasks), desc="파일 복사 중"))

if __name__ == "__main__":
    ultra_fast_sampling('dataset/Training', 'dataset_50k/Training', target_total=50000, bg_ratio=0.05)