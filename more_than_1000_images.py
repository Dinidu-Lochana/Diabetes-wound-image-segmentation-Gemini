import os
from PIL import Image

# Path to the folder containing images
folder_path = 'C:/24/Kainovation Technologies/Diabetes-wound-image-segmentation-Gemini/wound_dataset_groundingDino/images/train'

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(folder_path, filename)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > 1000 or height > 1000:
                    print(f"{filename} - Size: {width}x{height}")
        except Exception as e:
            print(f"Error opening {filename}: {e}")
