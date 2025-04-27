import cv2
import numpy as np
from ultralytics import YOLO

def detect_and_crop(image_path, model_path="C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/runs/detect/train10/weights/best.pt", save_path="clearest_date.jpg"):
    model = YOLO(model_path)
    results = model(image_path, conf=0.5)

    img = cv2.imread(image_path)
    best_crop = None
    best_score = -1
    non_overlapping_crops = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1

        # Ignore boxes that are too large or too small
        if width < 30 or height < 30 or width > 300 or height > 300:
            continue

        cropped = img[y1:y2, x1:x2]
        sharpness = measure_sharpness(cropped)
        noise = measure_noise(cropped)

        score = sharpness - (noise * 0.5)

        if score > best_score:
            best_score = score
            best_crop = cropped

        non_overlapping_crops.append((cropped, (x1, y1, x2, y2), score))

    if best_crop is not None:
        cv2.imwrite(save_path, best_crop)
        print(f"Saved clearest date to {save_path}!")

    # Draw circles around all detected date fruits
    draw_circles(img, non_overlapping_crops)


def measure_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def measure_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var


def draw_circles(image, crops):
    for crop, (x1, y1, x2, y2), score in crops:
        if score > 0:
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            radius = max((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.circle(image, center, radius, (0, 255, 0), 2)

    cv2.imwrite("output_with_circles.jpg", image)
    print("Saved output with circles as output_with_circles.jpg")


if __name__ == '__main__':
    detect_and_crop("C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/raw/test/6b23a97a.png")
