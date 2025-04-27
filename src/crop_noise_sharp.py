import cv2
import numpy as np
from ultralytics import YOLO
# STEP 4: Run inference and select the clearest date fruit
def detect_and_crop(image_path, model_path="C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/runs/detect/train10/weights/best.pt", save_path="clearest_date.jpg"):
    model = YOLO(model_path)  # Load trained YOLO model
    results = model(image_path, conf=0.5)  # Run inference

    img = cv2.imread(image_path)
    best_crop = None
    best_score = -1

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        cropped = img[y1:y2, x1:x2]  # Extract detected date fruit

        sharpness = measure_sharpness(cropped)
        noise = measure_noise(cropped)

        # SCORE = Sharpness - Noise (Higher is better)
        score = sharpness - (noise * 0.5)  # Adjust weight of noise penalty

        if score > best_score:
            best_score = score
            best_crop = cropped

    if best_crop is not None:
        cv2.imwrite(save_path, best_crop)
        print(f"Saved clearest date to {save_path}!")
# Measure noise using standard deviation of pixel intensities
def measure_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)
def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var
if __name__ == '__main__':
    detect_and_crop("C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/raw/test/7ff9f8f1.jpg")  # Detect and crop clearest date
