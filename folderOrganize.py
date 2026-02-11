import shutil
from pathlib import Path

def organize_dataset():
    for base_folder in ['dataset/Training', 'dataset/Validation']:
        src_path = Path(base_folder)
        if not src_path.exists():
            continue
            
        img_dir = src_path / 'images'
        lbl_dir = src_path / 'labels'
        
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Use list() to avoid issues with rglob while modifying the directory structure
        for file_path in list(src_path.rglob('*')):
            if file_path.is_file():
                # Skip files that are already inside the target folders
                if img_dir in file_path.parents or lbl_dir in file_path.parents:
                    continue
                    
                if file_path.suffix.lower() == '.jpg':
                    shutil.move(str(file_path), img_dir / file_path.name)
                elif file_path.suffix.lower() == '.json':
                    shutil.move(str(file_path), lbl_dir / file_path.name)

if __name__ == "__main__":
    organize_dataset()
