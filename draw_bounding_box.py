import cv2

# Image path
image_path = 'test1.jpg'

# Coordinates (x1, y1, x2, y2)
bounding_boxes = [
    [109, 233, 385, 670],
    [416, 225, 700, 558]
]

# Load the image
image = cv2.imread(image_path)

# Check if image is loaded properly
if image is None:
    print("Error: Image not found or invalid path!")
else:
    # Draw bounding boxes
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Display the image
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: Save the output
    # cv2.imwrite("output.jpg", image)
