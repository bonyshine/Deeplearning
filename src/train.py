from ultralytics import YOLO

if __name__ == '__main__':
    # Initialize YOLOv8 model
    model = YOLO("yolov8n.yaml")  # or yolov8s.yaml for better accuracy

    # Train the model
    model.train(data="dataset.yaml", epochs=50, imgsz=640, device=0)
