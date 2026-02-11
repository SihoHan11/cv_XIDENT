from pathlib import Path
import json

src_dir = Path('dataset/Training/annotations')
dst_dir = Path('dataset/Training/labels')

print(f"Counting files...")
src_files = sorted(list(src_dir.glob('*.json')))
print(f"Total source files: {len(src_files)}")

missing_sample = []
for i, p in enumerate(src_files):
    if not (dst_dir / (p.stem + '.txt')).exists():
        missing_sample.append(p)
    if len(missing_sample) >= 5:
        break

print(f"Missing sample: {missing_sample}")

for p in missing_sample:
    print(f"\n--- Content of {p.name} ---")
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
        try:
            bbox = data['ObjectInfo']['BoundingBox']
            print(f"BoundingBox keys: {list(bbox.keys())}")
            for k, v in bbox.items():
                print(f"  {k}: isVisible={v.get('isVisible')}, Opened={v.get('Opened')}")
        except Exception as e:
            print(f"Error reading JSON structure: {e}")
