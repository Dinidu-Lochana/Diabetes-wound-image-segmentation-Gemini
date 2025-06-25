import google.generativeai as genai
import os
from dotenv import load_dotenv
import os

load_dotenv()

# === STEP 1: Configure Gemini API ===
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# === STEP 2: Load image and create Gemini prompt ===
def load_image_for_gemini(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    return {
        "mime_type": "image/jpeg",  # or "image/png" if using PNG
        "data": image_bytes,
    }

# === STEP 3: Prompt Gemini with the image and instruction ===
def detect_diabetic_wound(image_data):
    model = genai.GenerativeModel("gemini-1.5-pro")

    prompt = (
        """
  Give the segmentation masks for the wounds.
  Output a JSON list of segmentation masks where each entry contains the 2D
  bounding box in the key "box_2d", the segmentation mask in key "mask", and
  the text label in the key "label". Use descriptive labels.
"""
    )

    response = model.generate_content([image_data, prompt])
    return response.text

# === STEP 4: Main ===
if __name__ == "__main__":
    image_path = "foot.jpg"  # <<< Replace with your actual image file
    try:
        image = load_image_for_gemini(image_path)
        result = detect_diabetic_wound(image)
        print("\n=== Diabetic Foot Wound Analysis ===\n")
        print(result)
    except Exception as e:
        print(f"Error: {e}")
