import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_file(file_info):
    file_path, dst_dir = file_info
    try:
        txt_path = dst_dir / f"{file_path.stem}.txt"
        if txt_path.exists():
            return "EXISTS"

        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        img_w = float(json_data['FileInfo']['Width'])
        img_h = float(json_data['FileInfo']['Height'])
        bbox_info = json_data['ObjectInfo']['BoundingBox']
        
        annotations = []
        mapping = {'Leye': (0, 1), 'Reye': (0, 1), 'Mouth': (2, 3)}
        
        for key, pos_info in bbox_info.items():
            if not pos_info.get('isVisible', False): continue
            
            # Position 값들이 문자열인 경우 대비하여 float 변환
            xmin, ymin, xmax, ymax = map(float, pos_info['Position'])
            
            x_c = (xmin + xmax) / 2.0 / img_w
            y_c = (ymin + ymax) / 2.0 / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h
            
            key_lower = key.lower()
            if 'eye' in key_lower:
                class_id = 0 if pos_info.get('Opened', False) else 1
            elif 'mouth' in key_lower:
                class_id = 2 if pos_info.get('Opened', False) else 3
            elif 'face' in key_lower:
                class_id = 4
            else: continue
                
            annotations.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            
        if annotations:
            with open(txt_path, 'w', encoding='utf-8') as tf:
                tf.write('\n'.join(annotations))
            return "CREATED"
        else:
            with open(txt_path, 'w', encoding='utf-8') as tf:
                tf.write('\n'.join(""))  # 객체가 없어서 빈 txt파일 생성
            return "EMPTY CREATED"
    except Exception:
        return "ERROR"

def run_conversion(mode, chunksize=500):
    base_path = Path(f'dataset/{mode}')
    src_dir = base_path / 'annotations'
    dst_dir = base_path / 'labels'
    dst_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(src_dir.glob('*.json'))
    if not json_files:
        print(f"[{mode}] 데이터 세트를 찾을 수 없습니다.")
        return

    print(f"\n[작업 시작] {mode} 데이터 세트: {len(json_files)}개")
    
    tasks = ((f, dst_dir) for f in json_files)
    
    results = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_single_file, tasks, chunksize=chunksize), 
            total=len(json_files), 
            desc=f"{mode} 변환 중"
        ))

    # 통계 계산
    stats = {
        "CREATED": results.count("CREATED"),
        "EXISTS": results.count("EXISTS"),
        "EMPTY CREATED": results.count("EMPTY CREATED"),
        "ERROR": results.count("ERROR")
    }
    
    print(f"\n[{mode} 작업 요약]")
    print(f"- 신규 생성: {stats['CREATED']}개")
    print(f"- 이미 존재: {stats['EXISTS']}개")
    print(f"- 객체 없음(빈 txt 생성): {stats['EMPTY CREATED']}개")
    if stats['ERROR'] > 0:
        print(f"- 오류 발생: {stats['ERROR']}개")

if __name__ == "__main__":
    for mode in ['Training', 'Validation']:
        run_conversion(mode)
    print("\n모든 작업 완료.")
