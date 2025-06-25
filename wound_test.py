from google.genai import types

with open('test1.jpg', 'rb') as f:
    image_bytes = f.read()

response = client.models.generate_content(
  model='gemini-2.0-flash',
  contents=[
    types.Part.from_bytes(
      data=image_bytes,
      mime_type='image/jpeg',
    ),
    'Caption this image.'
  ]
)

print(response.text)