import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def _copy_file_pair(args):
    img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir = args
    lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
    
    if lbl_path.exists():
        # 이미지와 라벨 복사
        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)

def create_pilot_dataset(src_base, dst_base, sample_ratio=0.1):
    src_img_dir = Path(src_base) / 'images'
    src_lbl_dir = Path(src_base) / 'labels'
    
    dst_img_dir = Path(dst_base) / 'images'
    dst_lbl_dir = Path(dst_base) / 'labels'
    
    # 대상 디렉토리 생성
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 확보
    all_images = list(src_img_dir.glob('*.jpg'))
    sample_size = int(len(all_images) * sample_ratio)
    
    # 무작위 샘플링
    pilot_images = random.sample(all_images, sample_size)
    
    print(f"파일럿 데이터 생성 중: {sample_size}개 추출 ({src_base})")
    
    # 병렬 처리를 위한 인자 리스트 생성
    tasks = [(img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir) for img_path in pilot_images]
    
    # 병렬 처리 실행 (청크 사이즈 10)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(_copy_file_pair, tasks, chunksize=10), total=len(tasks)))

if __name__ == "__main__":
    # Training과 Validation 각각 적용
    # create_pilot_dataset('dataset/Training', 'dataset_pilot/Training', sample_ratio=0.1)
    create_pilot_dataset('dataset/Validation', 'dataset_pilot/Validation', sample_ratio=0.1)
