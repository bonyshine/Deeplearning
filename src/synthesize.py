import cv2
import numpy as np
import os
import random
from PIL import Image

def load_images_all_classes(image_folder, annotation_folder):
    class_to_images = { "Sugaey": [], "Medjool": []}
    reverse_class_map = {6: "Sugaey", 1: "Medjool"}

    for filename in os.listdir(image_folder):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        img_path = os.path.join(image_folder, filename)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(annotation_path):
            continue

        # Just verify the image loads correctly
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Load annotations
        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        annotations_by_class = { 6: [], 1: []}
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            class_id = int(class_id)
            if class_id in annotations_by_class:
                annotations_by_class[class_id].append((class_id, x_center, y_center, width, height))

        for class_id, anns in annotations_by_class.items():
            if anns:
                label = reverse_class_map[class_id]
                # Just store filename instead of image array
                class_to_images[label].append((filename, anns, class_id))

    return class_to_images


def apply_random_transform(img):
    # Apply random horizontal and vertical flips
    if random.choice([True, False]):
        img = cv2.flip(img, 1)
    if random.choice([True, False]):
        img = cv2.flip(img, 0)
    # Apply random rotation: 0, 90, 180, or 270 degrees
    k = random.choice([0, 1, 2, 3])
    img = np.rot90(img, k)
    return img

def check_overlap(existing_boxes, new_box):
    for box in existing_boxes:
        x1, y1, x2, y2 = box
        nx1, ny1, nx2, ny2 = new_box
        if not (nx2 < x1 or nx1 > x2 or ny2 < y1 or ny1 > y2):
            return True
    return False

def get_random_selection_from_classes(class_to_images, total_images):
    """
    Randomly select `total_images` image file entries, one from each class (cycling through if needed).
    Returns a list of (image_filename, annotations, class_id) tuples.
    """
    selected = []
    class_names = list(class_to_images.keys())
    class_index = 0

    while len(selected) < total_images:
        class_name = class_names[class_index % len(class_names)]
        entries = class_to_images[class_name]
        if not entries:
            class_index += 1
            continue

        img_obj, annotations, class_id = random.choice(entries)
        # Save the original filename to reload the image later
        image_filename = os.path.basename(img_obj)
        selected.append((image_filename, annotations, class_id))

        class_index += 1

    return selected




def create_synthetic_image(selected_images, source_image_folder, output_path, annotation_folder, image_size=(640, 640)):
    synthetic_image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255
    synthetic_annotations = []
    placed_boxes = []

    h, w = image_size
    num_objects = len(selected_images)

    for img_filename, annotations, class_id in selected_images:
        img_path = os.path.join(source_image_folder, img_filename)
        print(f"Reading image: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        img_transformed = apply_random_transform(img)
        ih, iw = img_transformed.shape[:2]

        # Resize if too large to allow multiple images in canvas
        max_patch_height, max_patch_width = h // 2, w // 2
        if ih > max_patch_height or iw > max_patch_width:
            scale = min(max_patch_width / iw, max_patch_height / ih)
            new_w, new_h = int(iw * scale), int(ih * scale)
            img_transformed = cv2.resize(img_transformed, (new_w, new_h))
            ih, iw = new_h, new_w

        max_tries = 20
        for _ in range(max_tries):
            x_offset = random.randint(0, w - iw)
            y_offset = random.randint(0, h - ih)
            new_box = (x_offset, y_offset, x_offset + iw, y_offset + ih)

            if not check_overlap(placed_boxes, new_box):
                break
        else:
            print(f"Warning: Could not place {img_filename} without overlap after {max_tries} tries.")
            continue

        synthetic_image[y_offset:y_offset + ih, x_offset:x_offset + iw] = img_transformed
        placed_boxes.append(new_box)

        for annotation in annotations:
            _, x_c, y_c, bw, bh = annotation
            x_c_abs = x_c * iw + x_offset
            y_c_abs = y_c * ih + y_offset
            bw_abs = bw * iw
            bh_abs = bh * ih

            x_min = x_c_abs - bw_abs / 2
            y_min = y_c_abs - bh_abs / 2
            x_max = x_c_abs + bw_abs / 2
            y_max = y_c_abs + bh_abs / 2

            x_center_norm = ((x_min + x_max) / 2) / w
            y_center_norm = ((y_min + y_max) / 2) / h
            width_norm = (x_max - x_min) / w
            height_norm = (y_max - y_min) / h

            synthetic_annotations.append((class_id, x_center_norm, y_center_norm, width_norm, height_norm))

    synthetic_image = Image.fromarray(cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2RGB))
    synthetic_image.save(output_path)

    base_filename = os.path.splitext(os.path.basename(output_path))[0]
    annotation_file = os.path.join(annotation_folder, f"{base_filename}.txt")
    with open(annotation_file, 'w') as file:
        for ann in synthetic_annotations:
            file.write(" ".join(map(str, ann)) + "\n")

    print(f"✅ Saved synthetic image to {output_path}")
    print(f"✅ Saved synthetic annotations to {annotation_file}")



if __name__ == "__main__":
    image_folder = "C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/processed/images/train"
    annotation_folder = "C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/processed/labels/train"
    output_image_folder = "C:/Users/me.com/Documents/deepLearning/project/synthesize/synthetic-images"
    output_label_folder = "C:/Users/me.com/Documents/deepLearning/project/synthesize/synthetic-labels"

    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)


class_to_images = load_images_all_classes(image_folder, annotation_folder)

for i in range(50):
    n_images_total = random.choice([2, 3, 4])  # choose total number of images to sample
    selected_images = get_random_selection_from_classes(class_to_images, n_images_total)
    image_filename = f"synthetic_{i+300}.jpg"
    output_path = os.path.join(output_image_folder, image_filename)

    create_synthetic_image(
        selected_images,
        image_folder,
        output_path,
        output_label_folder,
        image_size=(640, 640)
    )