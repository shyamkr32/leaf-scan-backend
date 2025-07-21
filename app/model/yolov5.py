from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")

def predict_image(image_path: str):
    results = model.predict(image_path)

    if results and results[0].boxes.cls.numel() > 0:
        class_id = int(results[0].boxes.cls[0].item())
        confidence = float(results[0].boxes.conf[0].item())
        return class_id, confidence
    return None
