# 증강 / 샘플링 후 데이터 분포 확인import os
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor

def count_labels_in_file(file_path):
    """단일 라벨 파일에서 클래스 수를 세는 함수"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            classes = [line.split()[0] for line in lines if line.strip()]
            return Counter(classes)
    except Exception:
        return Counter()

def check_matching_images(base_dir):
    """이미지 파일 중 매칭되는 라벨 파일이 없는 파일 탐색"""
    img_dir = Path(base_dir) / 'images'
    lbl_dir = Path(base_dir) / 'labels'
    
    # 이미지 파일 목록 (jpg 기준)
    img_files = list(img_dir.glob('*.jpg')) + list((img_dir / 'augmented').glob('*.jpg'))
    
    orphan_images = []
    print(f"\nChecking image-label matching in {base_dir}...")
    
    for img_path in tqdm(img_files, desc=f"Matching {base_dir.name}"):
        # augmented 폴더 내의 파일인 경우 라벨도 augmented 폴더에서 찾음
        if 'augmented' in img_path.parts:
            lbl_path = lbl_dir / 'augmented' / f"{img_path.stem}.txt"
        else:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
        if not lbl_path.exists():
            orphan_images.append(img_path.name)
            
    return orphan_images

def check_distribution(base_dir, class_names):
    label_path = Path(base_dir) / 'labels'
    label_files = list(label_path.glob('*.txt')) + list(label_path.glob('augmented/*.txt'))
    
    print(f"\n[{base_dir.name}] Total label files found: {len(label_files)}")
    
    total_counter = Counter()
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(count_labels_in_file, label_files), 
                          total=len(label_files), 
                          desc=f"Analyzing {base_dir.name} labels"))
        
        for res in results:
            total_counter.update(res)

    # 결과 출력
    print(f"\n[ {base_dir.name} Class Distribution ]")
    labels = []
    counts = []
    for idx, name in enumerate(class_names):
        count = total_counter.get(str(idx), 0)
        print(f"Class {idx} ({name}): {count}")
        labels.append(f"{name}\n({idx})")
        counts.append(count)
        
    return labels, counts

def run_analysis():
    CLASS_NAMES = ['eye_opened', 'eye_closed', 'mouth_opened', 'mouth_closed', 'face']
    DATA_PATHS = [Path('dataset/Training'), Path('dataset/Validation')]
    
    all_orphan_info = {}

    for path in DATA_PATHS:
        if not path.exists():
            print(f"Warning: Directory not found - {path}")
            continue
            
        # 1. 클래스 분포 분석
        labels, counts = check_distribution(path, CLASS_NAMES)
        
        # 2. 매칭 확인
        orphans = check_matching_images(path)
        if orphans:
            all_orphan_info[path.name] = orphans
            
        # 3. 시각화 (개별 창으로 띄움)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color='skyblue')
        plt.title(f'{path.name} Class Distribution (Including Augmented Data)')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom')
        plt.tight_layout()

    # 최종 리포트 (매칭되지 않는 파일)
    if all_orphan_info:
        print("\n" + "="*50)
        print("!!! Orphan Images Found (No matching labels) !!!")
        for split, files in all_orphan_info.items():
            print(f"- {split}: {len(files)} files")
            for f in files:
                print(f"  > {f}")
        print("="*50)
    else:
        print("\nAll images have matching label files.")

    plt.show()

if __name__ == "__main__":
    run_analysis()
