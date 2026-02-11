import json
from pathlib import Path

def inspect_at(offset, count=5):
    src_dir = Path(r"c:\workspace\cv\dataset\Training\annotations")
    json_files = list(src_dir.glob("*.json"))
    
    total = len(json_files)
    print(f"Total files: {total}")
    
    end = min(offset + count, total)
    for i in range(offset, end):
        f = json_files[i]
        print(f"\n--- Index {i}: {f.name} ---")
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            # Print a bit of the structure
            print(f"Keys: {list(data.keys())}")
            if 'FileInfo' in data:
                print(f"FileInfo: {data['FileInfo']}")
                print(f"Width type: {type(data['FileInfo']['Width'])}, Height type: {type(data['FileInfo']['Height'])}")
            if 'ObjectInfo' in data and 'BoundingBox' in data['ObjectInfo']:
                bbox = data['ObjectInfo']['BoundingBox']
                print(f"BoundingBoxes: {list(bbox.keys())}")
                for k, v in bbox.items():
                    if v.get('isVisible'):
                        pos = v.get('Position')
                        print(f"  {k} Position: {pos} (types: {[type(x) for x in pos]})")
                        break # Just check one
            else:
                print("ObjectInfo or BoundingBox missing!")
        except Exception as e:
            print(f"Error reading: {e}")

if __name__ == "__main__":
    print("Checking near 100k:")
    inspect_at(100000)
    print("\nChecking near 200k:")
    inspect_at(200000)
