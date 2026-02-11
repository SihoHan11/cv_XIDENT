import os
from pathlib import Path
from tqdm.auto import tqdm

def find_orphan_images(base_dir):
    """이미지 파일 중 매칭되는 라벨 파일이 없는 파일 탐색"""
    img_dir = Path(base_dir) / 'images'
    lbl_dir = Path(base_dir) / 'labels'
    
    # 이미지 파일 목록 (jpg 기준)
    img_files = list(img_dir.glob('*.jpg'))
    aug_img_dir = img_dir / 'augmented'
    if aug_img_dir.exists():
        img_files.extend(list(aug_img_dir.glob('*.jpg')))
    
    orphan_images = []
    print(f"\nChecking image-label matching in {base_dir}...")
    
    for img_path in tqdm(img_files, desc=f"Matching {base_dir.name}"):
        # augmented 폴더 내의 파일인 경우 라벨도 augmented 폴더에서 찾음
        if 'augmented' in img_path.parts:
            lbl_path = lbl_dir / 'augmented' / f"{img_path.stem}.txt"
        else:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
        if not lbl_path.exists():
            orphan_images.append(img_path)
            
    return orphan_images

def run_label_matching_check():
    DATA_PATHS = [Path('dataset/Training'), Path('dataset/Validation')]
    all_orphan_info = {}

    for path in DATA_PATHS:
        if not path.exists():
            print(f"Warning: Directory not found - {path}")
            continue
            
        orphans = find_orphan_images(path)
        if orphans:
            all_orphan_info[path.name] = orphans

    # 최종 리포트 및 삭제 로직
    if all_orphan_info:
        print("\n" + "="*50)
        print("!!! Orphan Images Found (No matching labels) !!!")
        for split, files in all_orphan_info.items():
            print(f"- {split}: {len(files)} files")
            for f in files:
                print(f"  > {f.name}")
        print("="*50)
        
        confirm = input("\nDo you want to delete these orphan images? (y/n): ")
        if confirm.lower() == 'y':
            deleted_count = 0
            for split, files in all_orphan_info.items():
                for f in files:
                    if f.exists():
                        f.unlink()
                        deleted_count += 1
            print(f"Successfully deleted {deleted_count} orphan images.")
        else:
            print("Deletion cancelled.")
    else:
        print("\nAll images have matching label files.")

if __name__ == "__main__":
    run_label_matching_check()
