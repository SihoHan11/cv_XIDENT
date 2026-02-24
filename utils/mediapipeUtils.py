'''
import cv2
import mediapipe as mp

def mediapipeProcess(frame):
    # 1. MediaPipe 설정
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Bounding Box 좌표 저장용 변수
    bbox_coords = None
    pad_ratio = 0.3 # 패딩 비율 (30%)

    head_drop = False
    h, w, _ = frame.shape

    # AI 연산 수행
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # A. 고개 떨굼 감지 로직
            forehead = face_landmarks.landmark[10].y
            nose = face_landmarks.landmark[1].y
            chin = face_landmarks.landmark[152].y
            
            face_height = chin - forehead
            nose_pos_ratio = (nose - forehead) / face_height

            if nose_pos_ratio > 0.65: # 임계값
                head_drop = True
            else:
                head_drop = False

            # B. Bounding Box 좌표 계산
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            # 패딩 계산
            pw = int((max_x - min_x) * pad_ratio)
            ph = int((max_y - min_y) * pad_ratio)
            
            # 최종 좌표 저장 (이미지 경계 넘지 않게 clip 처리)
            start_x = max(0, min_x - pw)
            start_y = max(0, min_y - ph)
            end_x = min(w, max_x + pw)
            end_y = min(h, max_y + ph)
            
            # 좌표 저장 (이미지 경계 넘지 않도록 min, max 처리)
            start_x = max(0, min_x)
            start_y = max(0, min_y)
            end_x = min(w, max_x)
            end_y = min(h, max_y)

            # 좌표를 튜플로 저장해 둠
            bbox_coords = (start_x, start_y, end_x, end_y)
    else:
        # 얼굴을 놓쳤을 경우 None 반환
        return None

    # 패딩된 좌표 계산
    pw = int((max_x - min_x) * pad_ratio)
    ph = int((max_y - min_y) * pad_ratio)
    
    # 최종 좌표 저장 (이미지 경계 넘지 않게 clip 처리)
    start_x = max(0, min_x - pw)
    start_y = max(0, min_y - ph)
    end_x = min(w, max_x + pw)
    end_y = min(h, max_y + ph)

    # 크롭할 좌표 튜플로 저장
    crop_coords = (start_x, start_y, end_x, end_y)
    return [head_drop, bbox_coords, crop_coords]
    '''
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

'''class FaceProcessor:
    def __init__(self, model_path='face_landmarker.task'):
        # 1. MediaPipe Tasks 설정
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # 오류가 발생한 인자들을 제거하고 필수 인자만 넣습니다.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1  # 최대 얼굴 인식 수
            # output_face_blendshapes=False, <- 필요 없으면 삭제
            # output_face_transformation_matrixes=False <- 이 부분이 범인입니다!
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.pad_ratio = 0.3

    def process_frame(self, frame):
        h, w, _ = frame.shape
        # OpenCV BGR을 RGB로 변환 후 MediaPipe Image 객체 생성
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # AI 연산 수행
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        # 첫 번째 얼굴 데이터 추출
        face_landmarks = results.face_landmarks[0]

        # A. 고개 떨굼 감지 로직 (Landmark Index: 10=이마, 1=코끝, 152=턱끝)
        forehead = face_landmarks[10].y
        nose = face_landmarks[1].y
        chin = face_landmarks[152].y
        
        face_height = chin - forehead
        # 분모가 0이 되는 것을 방지
        nose_pos_ratio = (nose - forehead) / face_height if face_height != 0 else 0
        head_drop = nose_pos_ratio > 0.65

        # B. Bounding Box 및 Crop 좌표 계산
        # 모든 랜드마크로부터 min/max 좌표 추출
        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 원본 바운딩 박스 (정수형 변환)
        bbox_coords = (int(min_x), int(min_y), int(max_x), int(max_y))

        # 패딩 계산 (Crop용)
        pw = (max_x - min_x) * self.pad_ratio
        ph = (max_y - min_y) * self.pad_ratio

        start_x = int(max(0, min_x - pw))
        start_y = int(max(0, min_y - ph))
        end_x = int(min(w, max_x + pw))
        end_y = int(min(h, max_y + ph))
        
        crop_coords = (start_x, start_y, end_x, end_y)

        return [head_drop, bbox_coords, crop_coords]'''

class FaceProcessor:
    def __init__(self, model_path='face_landmarker.task'):
        # 1. MediaPipe Tasks 설정
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.pad_ratio = 0.3

    def process_frame(self, frame):
        # 시작 시간 측정
        start_time = time.time()
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]

        # A. 고개 떨굼 감지 로직
        forehead = face_landmarks[10].y
        nose = face_landmarks[1].y
        chin = face_landmarks[152].y
        
        face_height = chin - forehead
        nose_pos_ratio = (nose - forehead) / face_height if face_height != 0 else 0
        head_drop = nose_pos_ratio > 0.65

        # B. Bounding Box 및 Crop 좌표 계산
        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bbox_coords = (int(min_x), int(min_y), int(max_x), int(max_y))

        # 패딩 계산 (Crop용)
        pw = (max_x - min_x) * self.pad_ratio
        ph = (max_y - min_y) * self.pad_ratio

        start_x = int(max(0, min_x - pw))
        start_y = int(max(0, min_y - ph))
        end_x = int(min(w, max_x + pw))
        end_y = int(min(h, max_y + ph))
        
        # 크롭 좌표 튜플 생성
        crop_coords = (start_x, start_y, end_x, end_y)

        # C. 얼굴 기울기(Roll) 보정 및 정렬 (Face Alignment)
        left_eye_x = face_landmarks[33].x * w
        left_eye_y = face_landmarks[33].y * h
        right_eye_x = face_landmarks[263].x * w
        right_eye_y = face_landmarks[263].y * h

        # 기울기 각도 계산
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # 회전 중심점 계산
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        # 아핀 변환 행렬 생성 및 원본 이미지 회전
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

        # 정방향으로 회전된 프레임에서 안전하게 크롭 수행
        aligned_crop = rotated_frame[start_y:end_y, start_x:end_x]

        # 종료 시간 측정
        end_time = time.time()
        
        # 처리 시간 출력(ms)
        print('mediapipe processing time: '+ str(int((end_time - start_time) * 1000))+'ms')
        
        # 리턴 값 변경: 크롭 좌표(crop_coords)를 포함하여 리턴
        return [head_drop, bbox_coords, crop_coords, aligned_crop]

    def preprocess_image(self, frame, label, image_path='dataset_mediapipe/images/', label_path='dataset_mediapipe/labels/', cnt=0):
        start_time = time.time()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return -1

        face_landmarks = results.face_landmarks[0]

        # B. Bounding Box 및 Crop 좌표 계산
        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        bbox_coords = (int(min_x), int(min_y), int(max_x), int(max_y))

        pw = (max_x - min_x) * self.pad_ratio
        ph = (max_y - min_y) * self.pad_ratio

        start_x = int(max(0, min_x - pw))
        start_y = int(max(0, min_y - ph))
        end_x = int(min(w, max_x + pw))
        end_y = int(min(h, max_y + ph))

        # C. 얼굴 기울기(Roll) 보정 및 정렬 (Face Alignment)
        left_eye_x = face_landmarks[33].x * w
        left_eye_y = face_landmarks[33].y * h
        right_eye_x = face_landmarks[263].x * w
        right_eye_y = face_landmarks[263].y * h

        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

        aligned_crop = rotated_frame[start_y:end_y, start_x:end_x]
        crop_h, crop_w = aligned_crop.shape[:2]

        # D. YOLO 라벨 좌표 회전 및 크롭 동기화
        new_labels = []
        if label:  
            lines = label.strip().split('\n')
            for line in lines:
                if not line.strip(): continue
                parts = line.strip().split()
                class_id = parts[0]
                cx_norm, cy_norm, bw_norm, bh_norm = map(float, parts[1:])

                # 1. 정규화 좌표 -> 원본 이미지 절대 좌표
                cx_abs = cx_norm * w
                cy_abs = cy_norm * h
                bw_abs = bw_norm * w
                bh_abs = bh_norm * h

                # 2. 바운딩 박스 네 모서리 좌표 추출
                corners = np.array([
                    [cx_abs - bw_abs/2, cy_abs - bh_abs/2], # Top-Left
                    [cx_abs + bw_abs/2, cy_abs - bh_abs/2], # Top-Right
                    [cx_abs - bw_abs/2, cy_abs + bh_abs/2], # Bottom-Left
                    [cx_abs + bw_abs/2, cy_abs + bh_abs/2]  # Bottom-Right
                ])

                # 3. 아핀 변환 행렬(M)을 적용하여 모서리 회전
                ones = np.ones(shape=(len(corners), 1))
                corners_ones = np.hstack([corners, ones])
                rotated_corners = M.dot(corners_ones.T).T

                # 4. 회전된 모서리들을 포함하는 새로운 축 정렬 바운딩 박스 계산
                new_min_x = np.min(rotated_corners[:, 0])
                new_max_x = np.max(rotated_corners[:, 0])
                new_min_y = np.min(rotated_corners[:, 1])
                new_max_y = np.max(rotated_corners[:, 1])

                # 5. 크롭 좌표계로 평행 이동
                new_min_x -= start_x
                new_max_x -= start_x
                new_min_y -= start_y
                new_max_y -= start_y

                # 6. 크롭 이미지 영역을 벗어난 좌표 자르기 (Clipping)
                new_min_x = np.clip(new_min_x, 0, crop_w)
                new_max_x = np.clip(new_max_x, 0, crop_w)
                new_min_y = np.clip(new_min_y, 0, crop_h)
                new_max_y = np.clip(new_max_y, 0, crop_h)

                # 박스가 영역 밖으로 완전히 사라진 경우 예외 처리
                if new_max_x <= new_min_x or new_max_y <= new_min_y:
                    continue

                # 7. 크롭 이미지 크기에 맞춰 다시 정규화 (0~1)
                new_cx = ((new_min_x + new_max_x) / 2) / crop_w
                new_cy = ((new_min_y + new_max_y) / 2) / crop_h
                new_bw = (new_max_x - new_min_x) / crop_w
                new_bh = (new_max_y - new_min_y) / crop_h

                new_labels.append(f"{class_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

        # E. 이미지 및 변환된 라벨 저장 (경로 변수 통일)
        cv2.imwrite(f"{image_path}crop{cnt}.jpg", aligned_crop)
        
        if new_labels:
            with open(f"{label_path}crop{cnt}.txt", "w") as f:
                f.write("\n".join(new_labels))

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        return processing_time

# 얼굴 박스 그리기
def draw_face_box(frame, bbox_coords, drop_head):
    if bbox_coords is not None:
        x1, y1, x2, y2 = bbox_coords
        if drop_head:
            color = (0, 0, 255) # Red
            thickness = 3
        else:
            color = (255, 255, 255) # White
            thickness = 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame