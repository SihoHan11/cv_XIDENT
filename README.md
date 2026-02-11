# WIP-cv_XIDENT: 실시간 졸음운전 감지 시스템

> **프로젝트 목표:** Computer Vision을 활용한 고속/고정밀 졸음 판별 및 경보 시스템 개발

---

### 데이터 및 모델 정보
- **사용 데이터:** [AI Hub - 운전자 상태 정보 영상 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=173)
- **사용 모델:** [Ultralytics YOLO26](https://docs.ultralytics.com/ko/models/yolo26/) (YOLO26n ~ YOLO26m 중 선택 예정)

---

### 데이터 전처리 현황 (WIP)

- [x] **디렉토리 구조 정리** (`folderOrganize.py`)
- [x] **데이터 클래스 정의**
  - `open_eye`, `close_eye`, `mouth_open`, `mouth_close`, `face`
- [x] **레이블 데이터 정규화** (`jsonNormalizeYolo.py`, `jsonNormalizeYolo.ipynb`)
- [x] **데이터 분포 확인** (`dataVisualization.ipynb`)
- [ ] **데이터 불균형 문제 처리** (`dataBalancingPro.py`) *진행 중*
- [ ] **학습/테스트 데이터셋 분리**
- [ ] **YOLO 학습을 위한 YAML 설정 파일 생성**
- [ ] **최종 데이터셋 검증 및 무결성 확인**


### 모델 학습 및 평가 (Planned)

- [ ] **Pilot Training 및 타당성 검토 (Feasibility Study)** (10% 샘플링 데이터를 활용한 모델 수렴성 및 학습 가능성 조기 검증)
- [ ] **YOLO 모델 학습** (Pre-trained 가중치 활용 및 전이 학습)
- [ ] **성능 지표 측정 및 분석** (mAP50, Precision, Recall, F1-score)
- [ ] **하이퍼파라미터 튜닝** (Learning Rate, Batch Size, Augmentation 설정 최적화)
- [ ] **모델 경량화 및 최적화** (TensorRT/ONNX 변환을 통한 추론 속도 개선)
- [ ] **실시간 데모 및 경보 시스템 통합 테스트**
