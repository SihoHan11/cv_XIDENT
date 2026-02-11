import json
from pathlib import Path
import traceback

def process_single_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        print("JSON loaded successfully.")
        
        # Checking keys
        if 'FileInfo' not in json_data:
            print("ERROR: 'FileInfo' key missing")
            print("Keys available:", json_data.keys())
            return
            
        img_w = json_data['FileInfo']['Width']
        img_h = json_data['FileInfo']['Height']
        
        if 'ObjectInfo' not in json_data:
            print("ERROR: 'ObjectInfo' key missing")
            return
            
        bbox_info = json_data['ObjectInfo']['BoundingBox']
        
        print(f"Image size: {img_w}x{img_h}")
        print(f"Found {len(bbox_info)} bounding boxes.")
        
        for key, pos_info in bbox_info.items():
            print(f"Processing key: {key}")
            # The original code does: if not pos_info.get('isVisible', False): continue
            # Let's check what's in pos_info
            print(f"Pos info: {pos_info}")
            
            if not pos_info.get('isVisible', False): 
                print(f"Skipping {key} because isVisible is False or missing.")
                continue
            
            if 'Position' not in pos_info:
                print(f"ERROR: 'Position' missing in {key}")
                continue
                
            xmin, ymin, xmax, ymax = pos_info['Position']
            print(f"Position: {xmin}, {ymin}, {xmax}, {ymax}")

    except Exception as e:
        print(f"Exception caught: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    src_dir = Path(r"c:\workspace\cv\dataset\Training\annotations")
    json_files = list(src_dir.glob("*.json"))
    
    total = len(json_files)
    print(f"Total files found: {total}")
    
    # Check 1000 files scattered across the list
    indices = [i for i in range(0, total, max(1, total // 1000))]
    
    error_counts = {}
    processed = 0
    errors = 0
    
    for i in indices:
        f = json_files[i]
        processed += 1
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                # Try to detect encoding if utf-8 fails
                try:
                    data = json.load(jf)
                except UnicodeDecodeError:
                    # Try cp949
                    with open(f, 'r', encoding='cp949') as jf2:
                        data = json.load(jf2)
                        # If this works, we know it's an encoding issue
                        raise ValueError("Encoding issue: cp949")
            
            # Simple simulation of the original script's logic
            if 'FileInfo' not in data:
                raise KeyError("FileInfo missing")
            w = data['FileInfo']['Width']
            h = data['FileInfo']['Height']
            
            if 'ObjectInfo' not in data:
                raise KeyError("ObjectInfo missing")
            bbox = data['ObjectInfo']['BoundingBox']
            
            for k, v in bbox.items():
                if v.get('isVisible'):
                    if 'Position' not in v:
                        raise KeyError(f"Position missing in {k}")
                    pos = v['Position']
                    if len(pos) != 4:
                        raise ValueError(f"Position of {k} has {len(pos)} elements")
        except Exception as e:
            errors += 1
            err_type = type(e).__name__
            err_msg = str(e)
            key = f"{err_type}: {err_msg}"
            error_counts[key] = error_counts.get(key, 0) + 1
            if errors <= 10:
                print(f"Error in index {i}, file {f.name}:")
                print(f"Error: {key}")

    print(f"\nScan complete. Processed: {processed}, Errors: {errors}")
    for err, count in error_counts.items():
        print(f"{err}: {count}")
