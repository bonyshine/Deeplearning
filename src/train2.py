import cv2
import sys
import os
import numpy as np
# Set the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ultralytics import YOLO

# Load pretrained YOLOv8 small model (recommended over training from scratch)
model = YOLO("yolov8n.pt")  # You can try yolov8m.pt or yolov8n.pt too

if __name__ == '__main__':
    model.train(
        data="dataset.yaml",   # Path to your dataset config file
        epochs=1,
        imgsz=640,
        batch=16,
        device=0,              # GPU (set to 'cpu' if needed)
        workers=4,             # Adjust based on your CPU cores

        # Lighter and safer augmentations
        augment=True,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        mosaic=0.5,
        flipud=0.1,
        fliplr=0.3,
        degrees=5,
        translate=0.05,
        scale=0.3,
        shear=0,

        # Optional: Save best weights only
        save=True,
        save_period=-1,  # Only save best.pt

        # Optional: Verbose output and logging
        verbose=True
    )
