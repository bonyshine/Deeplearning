import os
import random

# Define paths
base_dir1 = "C:/Users/me.com/Documents/deepLearning/project/synthesize"
raw_images_dir = os.path.join(base_dir1,  "synthetic-imagesCopy")  # Folder containing original images
raw_labels_dir = os.path.join(base_dir1, "synthetic-labelsCopy")  # Folder containing annotation txt files


# Define paths
base_dir = "C:/Users/me.com/Documents/deepLearning/project/date-fruit-classification/data/processed"
train_images_dir = os.path.join(base_dir, "images", "trainSyn")
val_images_dir = os.path.join(base_dir,  "images", "valSyn")
train_labels_dir = os.path.join(base_dir,  "labels", "trainSyn")
val_labels_dir = os.path.join(base_dir, "labels", "valSyn")



# List all images and labels
image_files = [f for f in os.listdir(raw_images_dir) if f.endswith((".jpg", ".png"))]
label_files = [f for f in os.listdir(raw_labels_dir) if f.endswith(".txt")]

print("✅ Total Images:", len(image_files))
print("✅ Total Labels:", len(label_files))

# Check first 5 files
print("\nFirst 5 images:", image_files[:5])
print("First 5 labels:", label_files[:5])

# Get list of images
image_files = [f for f in os.listdir(raw_images_dir) if f.endswith((".jpg", ".png"))]  # Adjust extensions if needed

# Check if images exist
print("✅ Total Images Found:", len(image_files))
print("First 5 Images:", image_files[:5])  # Print first 5 images to confirm

# Shuffle image files
random.shuffle(image_files)

# Split dataset (80% train, 20% val)
train_size = int(0.8 * len(image_files))
train_files = image_files[:train_size]
val_files = image_files[train_size:]

print("\n✅ Train Set Size:", len(train_files))
print("✅ Validation Set Size:", len(val_files))

# Check if every image has a corresponding label
missing_labels = [
    f for f in train_files + val_files if not os.path.exists(os.path.join(raw_labels_dir, f.rsplit('.', 1)[0] + ".txt"))
]

if missing_labels:
    print("\n⚠️ WARNING: Some images are missing labels!")
    print(missing_labels[:5])  # Show first 5 missing labels
else:
    print("\n✅ All images have corresponding labels!")
import shutil

# Function to move files safely
def move_file(src_folder, dest_folder, file_name):
    src_path = os.path.join(src_folder, file_name)
    dest_path = os.path.join(dest_folder, file_name)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"⚠️ File missing: {src_path}")

# Move train images & labels
for file in train_files:
    move_file(raw_images_dir, train_images_dir, file)
    label_file = file.rsplit(".", 1)[0] + ".txt"
    move_file(raw_labels_dir, train_labels_dir, label_file)

# Move val images & labels
for file in val_files:
    move_file(raw_images_dir, val_images_dir, file)
    label_file = file.rsplit(".", 1)[0] + ".txt"
    move_file(raw_labels_dir, val_labels_dir, label_file)



print("✅ Dataset successfully split into train, val!")
