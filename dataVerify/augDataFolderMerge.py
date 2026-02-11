# 증강된 데이터 폴더를 메인 폴더로 이동
import os
import shutil
'''
base_dir = 'dataset/Training'
for sub in ['labels', 'images']:
    aug_path = os.path.join(base_dir, sub, 'augmented')
    if os.path.exists(aug_path):
        dest_path = os.path.join(base_dir, sub)
        for filename in os.listdir(aug_path):
            shutil.move(os.path.join(aug_path, filename), os.path.join(dest_path, filename))
        # 빈 폴더 삭제
        os.rmdir(aug_path)
'''
base_dir = 'dataset/Validation'
for sub in ['labels', 'images']:
    aug_path = os.path.join(base_dir, sub, 'augmented')
    if os.path.exists(aug_path):
        dest_path = os.path.join(base_dir, sub)
        for filename in os.listdir(aug_path):
            shutil.move(os.path.join(aug_path, filename), os.path.join(dest_path, filename))
        # 빈 폴더 삭제
        os.rmdir(aug_path)