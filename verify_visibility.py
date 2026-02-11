from pathlib import Path
import json

src_dir = Path('dataset/Training/annotations')
dst_dir = Path('dataset/Training/labels')

src_files = sorted(list(src_dir.glob('*.json')))

print(f"Checking files from index 200,000 to 201,000...")
processed = 0
with_objects = 0
without_objects = 0

for i in range(200000, min(201000, len(src_files))):
    p = src_files[i]
    processed += 1
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
        bbox = data['ObjectInfo']['BoundingBox']
        visible = any(v.get('isVisible', False) for v in bbox.values())
        if visible:
            with_objects += 1
        else:
            without_objects += 1

print(f"Processed: {processed}")
print(f"With visible objects: {with_objects}")
print(f"Without visible objects: {without_objects}")
