import sys
import os
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import logging
import cv2


# Path to local ultralytics repo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))
sys.path.insert(0, project_root)
from ultralytics import YOLO
# Load custom config
yaml_path = os.path.join(project_root, 'ultralytics', 'cfg', 'models', '11', 'yolo11n.yaml')
model = YOLO(yaml_path)

if __name__ == '__main__':
    model.train(
        data="C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=4,               # fit for 4GB GPU
        workers=2,
        device=0,
        cache=False,           # synthetic data already fast to load

        # Optimizer
        lr0=0.005,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0002,

        # Warmup
        warmup_epochs=2.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # 🔥 Minimal augmentations (your data is already augmented)
        degrees=0.0,
        translate=0.02,        # tiny shift if any
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.3,            # optional
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,            # synthetic data doesn't need mosaic
        mixup=0.0,
        copy_paste=0.0,

        # Other
        verbose=True,
        patience=20,
        close_mosaic=0,        # no mosaic anyway
        name='cbfocal-yolo11n-synth',
    )
