import cv2
import numpy as np
from ultralytics import YOLO

# Load your pre-trained model for fine-tuning
model = YOLO("C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/runs/detect/train10/weights/best.pt")

if __name__ == '__main__':
    model.train(
        data="dataset.yaml",  # Dataset config
        epochs=100,           # Continue training for more epochs
        imgsz=640,            # High resolution for better detection
        batch=16,             # Adjust batch size based on GPU memory
        device=0,             # Use GPU if available
        augment=True,         # Data augmentation

        # Fine-tuning settings
       
        lr0=0.001,            # Initial learning rate (smaller for fine-tuning)
        lrf=0.0001,           # Final learning rate
        patience=10,          # Early stopping: Stops if no improvement in 10 epochs

        # Data Augmentation: Improves robustness
        hsv_h=0.015,          
        hsv_s=0.7,           
        hsv_v=0.4,           
        mosaic=1.0,          
        flipud=0.5,          
        fliplr=0.5,          
        degrees=10,          
        translate=0.1,       
        scale=0.5,           
        shear=2,             

        # Checkpoints and saving settings
        save_period=5,        # Save every 5 epochs
        exist_ok=True,        # Overwrite previous runs

        # Mixed data support (if real data available)
        rect=True,            # Maintain aspect ratio of images
        workers=4,            # Number of data loading workers
    )
