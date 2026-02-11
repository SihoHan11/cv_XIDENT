# 라벨 파일이 비어있는 이미지 확인
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def is_label_empty(file_path):
    try:
        return os.path.getsize(file_path) == 0
    except OSError:
        return False

def count_empty_labels(label_dir="dataset/Training/labels"):
    # os.scandir를 사용하여 파일 목록을 더 빠르게 가져옴
    label_files = [entry.path for entry in os.scandir(label_dir) if entry.is_file() and entry.name.endswith('.txt')]
    
    if not label_files:
        print("No label files found.")
        return 0

    # 프로세스 간 통신 오버헤드를 줄이기 위해 chunksize 설정
    # 작업이 매우 가벼우므로 큰 chunksize가 유리함
    chunksize = max(1, len(label_files) // (os.cpu_count() * 10))
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(is_label_empty, label_files, chunksize=chunksize), 
            total=len(label_files), 
            desc="Checking Labels"
        ))
    
    empty_count = sum(results)
    print(f"Number of empty label files: {empty_count}")
    return empty_count

if __name__ == "__main__":
    count_empty_labels()
