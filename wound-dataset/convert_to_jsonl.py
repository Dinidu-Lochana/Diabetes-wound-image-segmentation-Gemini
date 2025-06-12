import json

# Load COCO annotation file
with open("_annotations.coco.json", "r") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}
annotations_by_image = {}
for ann in coco["annotations"]:
    annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# Map category IDs to names
category_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

jsonl_lines = []

for image_id, img in images.items():
    anns = annotations_by_image.get(image_id, [])
    wounds = []
    for ann in anns:
        bbox = ann["bbox"]  # COCO bbox: [x, y, width, height]
        # Convert to Gemini bbox: [ymin, xmin, ymax, xmax]
        ymin = bbox[1]
        xmin = bbox[0]
        ymax = bbox[1] + bbox[3]
        xmax = bbox[0] + bbox[2]
        wounds.append({
            "label": category_map.get(ann["category_id"], "wound"),
            "box_2d": [ymin, xmin, ymax, xmax]
        })

    output = {"wounds": wounds}

    example = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/jpeg",  
                            "fileUri": f"gs://your-bucket/wounds/{img['file_name']}"
                        }
                    },
                    {
                        "text": "Identify and localize all wounds in this image."
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": json.dumps(output)
                    }
                ]
            }
        ]
    }

    jsonl_lines.append(json.dumps(example))

# Save to JSONL file
with open("wound_train.jsonl", "w") as f:
    for line in jsonl_lines:
        f.write(line + "\n")

print("Conversion complete! JSONL saved as wound_train.jsonl")
