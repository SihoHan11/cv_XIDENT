import os
import random
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

def move_task_worker(task):
    """파일 이동을 담당하는 개별 프로세스 유닛"""
    src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, stem = task
    try:
        # 이미지와 라벨을 세트로 이동
        img_name = f"{stem}.jpg" # 네 이미지 확장자에 맞춰 확인해
        lbl_name = f"{stem}.txt"
        
        if (src_img_dir / img_name).exists():
            shutil.move(src_img_dir / img_name, dst_img_dir / img_name)
        if (src_lbl_dir / lbl_name).exists():
            shutil.move(src_lbl_dir / lbl_name, dst_lbl_dir / lbl_name)
        return True
    except Exception as e:
        return False

def split_val_test(val_base, test_base, split_ratio=0.5):
    val_base, test_base = Path(val_base), Path(test_base)
    val_img, val_lbl = val_base / 'images', val_base / 'labels'
    test_img, test_lbl = test_base / 'images', test_base / 'labels'

    # Test 폴더 생성
    for d in [test_img, test_lbl]: d.mkdir(parents=True, exist_ok=True)

    # Validation 라벨 파일 목록 확보
    print(f"\n[분석] {val_base} 데이터 스캔 중...")
    all_stems = [p.stem for p in val_lbl.glob('*.txt')]
    random.shuffle(all_stems)
    
    # 절반 선별
    split_idx = int(len(all_stems) * split_ratio)
    test_stems = all_stems[:split_idx]
    
    print(f"[실행] {len(test_stems)}개 파일을 Test 세트로 이동 중 (병렬 처리)...")
    
    move_tasks = [(val_img, val_lbl, test_img, test_lbl, s) for s in test_stems]
    
    with ProcessPoolExecutor() as executor:
        # 이동 작업은 I/O 작업이므로 코어를 충분히 활용해
        list(tqdm(executor.map(move_task_worker, move_tasks, chunksize=100), 
                  total=len(move_tasks), desc="데이터 분리 중"))

if __name__ == "__main__":
    # 기존 Validation 경로와 새로 만들 Test 경로를 지정해
    split_val_test('dataset/Validation', 'dataset/Test', split_ratio=0.5)