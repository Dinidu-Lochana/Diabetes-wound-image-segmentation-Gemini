import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Set your API key
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Load image
image_path = "test1.jpg"
image = Image.open(image_path).convert("RGB")

# Prompt
prompt = """
Give the segmentation masks for the wooden and glass items.
  Output a JSON list of segmentation masks where each entry contains the 2D
  bounding box in the key "box_2d", the segmentation mask in key "mask", and
  the text label in the key "label". Use descriptive labels.
"""

# Create model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Generate content using direct [image, text] prompt
response = model.generate_content([image, prompt])

# Print raw response text
print("MODEL RAW RESPONSE:\n", response.text)

# Try parsing the response as JSON
try:
    result = json.loads(response.text)
except json.JSONDecodeError as e:
    print("JSON Decode Error:", e)
    exit(1)

# Annotate image
draw = ImageDraw.Draw(image)

# Load font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Draw segmentation and labels
for item in result:
    label = item["label"]
    box = item["box_2d"]
    mask = item["mask"][0]

    draw.rectangle(
        [(box["x_min"], box["y_min"]), (box["x_max"], box["y_max"])],
        outline="red",
        width=2
    )
    draw.polygon(mask, outline="blue")
    draw.text((box["x_min"], box["y_min"] - 20), label, fill="yellow", font=font)

# Save and show
output_path = "annotated_test2.jpg"
image.save(output_path)
image.show()

print(f"Segmentation saved to {output_path}")
