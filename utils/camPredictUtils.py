import cv2
import numpy as np

def calculate_iou(box1, box2):
    """
    두 박스(box1, box2)의 IoU를 계산하는 함수
    박스 형식: [x1, y1, x2, y2] (좌상단 x, y, 우하단 x, y)
    """
    # 1. 겹치는 영역의 좌표 계산
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 2. 겹치는 영역이 없으면 IoU 0 반환
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 3. 겹치는 영역의 넓이 계산 (Intersection)
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 4. 각 박스의 넓이 계산
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 5. 합집합의 넓이 계산 (Union = A + B - Intersection)
    union_area = float(box1_area + box2_area - intersection_area)

    # 6. IoU 계산
    iou = intersection_area / union_area
    return iou

def filter_overlapping_parts(results, iou_threshold=0.3):
    boxes = results[0].boxes
    data = boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
    
    # 0, 1(눈) 그룹과 2, 3(입) 그룹 인덱스 정의
    eye_indices = [i for i, x in enumerate(data) if x[5] in [0, 1]]
    mouth_indices = [i for i, x in enumerate(data) if x[5] in [2, 3]]
    
    keep_indices = []

    def get_best_idx(indices, weight_map={1: 1.05, 2: 1.2}):
        """
        weight_map: {class_id: weight_value}
        닫힌 눈(1)과 열린 입(2)에 가중치를 부여
        """
        if not indices: return []
        
        # 가중치가 적용된 신뢰도를 기준으로 정렬
        def get_weighted_conf(i):
            cls = int(data[i][5])
            conf = data[i][4]
            return conf * weight_map.get(cls, 1.0)

        sorted_indices = sorted(indices, key=get_weighted_conf, reverse=True)
        
        valid = []
        used = set()
        
        for i in sorted_indices:
            if i in used: continue
            valid.append(i)
            for j in sorted_indices:
                if i == j or j in used: continue
                if calculate_iou(data[i][:4], data[j][:4]) > iou_threshold:
                    used.add(j)
        return valid

    # 눈과 입 각각에 대해 최선의 박스만 추출
    keep_indices.extend(get_best_idx(eye_indices))
    keep_indices.extend(get_best_idx(mouth_indices))
    
    # 얼굴(class 4)은 그대로 유지
    # face_indices = [i for i, x in enumerate(data) if x[5] == 4]
    # keep_indices.extend(face_indices)
    
    return data[keep_indices]

def draw_filtered_results(frame, filtered_data, class_names, crop_coords):
    """
    filtered_data: [[x1, y1, x2, y2, conf, cls], ...] 형태의 numpy array
    """
    for box in filtered_data:
        x1, y1, x2, y2, conf, cls = box
        cx1, cy1 = crop_coords[:2]
        # OpenCV 좌표는 정수여야 하니까 형변환 잊지 마
        p1, p2 = (int(x1+cx1), int(y1+cy1)), (int(x2+cx1), int(y2+cy1))
        
        # 클래스별 색상 지정 (BGR 순서)
        if cls in [1, 2]: # 이상 상태
            color = (0, 0, 255) # Red
            thickness = 3
        elif cls in [0, 3]: # 정상 상태
            color = (0, 255, 0) # Green
            thickness = 2
            
        # 박스 그리기
        cv2.rectangle(frame, p1, p2, color, thickness)
        
        # 라벨 텍스트 (클래스명 + 신뢰도)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        cv2.putText(frame, label, (p1[0], p1[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
    return frame

# 하품 감지 함수
def isYawning(box, bbox_coords):
    # 입의 종횡비 계산 (세로/가로)
    x1, y1, x2, y2 = box[:4]
    width = x2 - x1
    height = y2 - y1
    
    if width == 0:
        return False
        
    # 종횡비가 0.6 이상이면 하품 상태로 판단
    ratio = height / width
    if ratio >= 0.6:
        return True
    
    # 얼굴 면적 대비 입의 면적비 계산
    face_area = (bbox_coords[2] - bbox_coords[0]) * (bbox_coords[3] - bbox_coords[1])
    mouth_area = (x2 - x1) * (y2 - y1)
    
    # 0.1 이상이면 하품 상대로 판단
    if mouth_area / face_area >= 0.1:
        return True
    
    return False