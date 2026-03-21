import os
import json
from PIL import Image

# ==== CONFIG ====
DATASET_PATH = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/Data/Video/combined_gunsnknifes"
SPLITS = ["train", "val"]
CLASS_NAMES = ['pistol', 'knife']
# =================

def convert_split(split):
    images_path = os.path.join(DATASET_PATH, split, "images")
    labels_path = os.path.join(DATASET_PATH, split, "labels")

    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for i, name in enumerate(CLASS_NAMES):
        coco["categories"].append({
            "id": i,
            "name": name
        })

    annotation_id = 0
    image_id = 0

    for image_file in os.listdir(images_path):
        if not image_file.endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = Image.open(image_path)
        width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                class_id, x_center, y_center, w, h = map(float, line.strip().split())

                x = (x_center - w / 2) * width
                y = (y_center - h / 2) * height
                w = w * width
                h = h * height

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    output_path = os.path.join(DATASET_PATH, split, "_annotations.coco.json")

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"✅ Converted {split} → COCO format")


for split in SPLITS:
    convert_split(split)

print("🎉 Conversion Complete")