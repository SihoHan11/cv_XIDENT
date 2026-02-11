from ultralytics import YOLO

# 가장 가벼운 모델 로드 (파일럿 검증용)
model = YOLO('yolo26n.pt')

# 학습 실행
train_result = model.train(
    data='data_pilot.yaml',   # 파일럿 데이터 경로가 적힌 yaml
    epochs=10,                # 검증용이므로 짧게 설정
    imgsz=640,
    batch=8,                  # CPU 부하를 고려해 작은 값 설정
    device='cpu',             # CPU 강제 사용 설정
    workers=10,                 # CPU 코어 수에 맞춰 조절
    dfl=2.0                # '맞추기 쉬운 샘플'의 비중을 줄이고 '어려운 샘플'에 집중하게 만드는 방식
)

metrics = model.val()
'''
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx") '''