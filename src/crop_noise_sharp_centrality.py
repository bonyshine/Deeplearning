import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_crop(image_path, model_path="C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/runs/detect/train12/weights/best.pt", save_path="clearest_date2.jpg"):
    model = YOLO(model_path)  # Load trained YOLO model
    results = model(image_path, conf=0.5)  # Run inference

    img = cv2.imread(image_path)
    height, width, _ = img.shape
    best_crop = None
    best_score = -1

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        cropped = img[y1:y2, x1:x2]  # Extract detected date fruit

        sharpness = measure_sharpness(cropped)
        noise = measure_noise(cropped)
        centrality = measure_centrality(x1, y1, x2, y2, width, height)
        size_score = measure_size(x1, y1, x2, y2, width, height)

        # Combined score calculation
        score = sharpness - (noise * 0.5) + (centrality * 0.3) + (size_score * 0.2)

        if score > best_score:
            best_score = score
            best_crop = cropped

    if best_crop is not None:
        cv2.imwrite(save_path, best_crop)
        print(f"Saved clearest date to {save_path}!")

def measure_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def measure_centrality(x1, y1, x2, y2, img_width, img_height):
    # Calculate center of the bounding box
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    img_center_x = img_width / 2
    img_center_y = img_height / 2

    # Euclidean distance from image center, normalized by image dimensions
    distance = np.sqrt((bbox_center_x - img_center_x) ** 2 + (bbox_center_y - img_center_y) ** 2)
    max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
    
    # Centrality score (higher is better, so invert the distance)
    return 1 - (distance / max_distance)

def measure_size(x1, y1, x2, y2, img_width, img_height):
    # Calculate bounding box area
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    
    # Calculate area ratio (should be neither too small nor too large)
    area_ratio = bbox_area / img_area
    if 0.01 <= area_ratio <= 0.1:  # Assuming ideal area range for full fruit detection
        return 1.0
    else:
        return max(0.0, 1 - abs(area_ratio - 0.05) / 0.05)  # Penalize if outside ideal range

if __name__ == '__main__':
    detect_and_crop("C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/raw/test/0fc62b58.jpg")
