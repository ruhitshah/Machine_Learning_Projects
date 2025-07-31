import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can choose other models like 'yolov8s.pt' or 'yolov8m.pt'

# Initialize webcam capture (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the live stream.")

while True:
    # Capture frame-by-frame from the webcam feed
    ret, frame = cap.read()

    # If frame is not captured properly, break the loop
    if not ret:
        print("Failed to capture frame")
        break

    # Perform object detection on the current frame
    results = model(frame)  # YOLOv8 processes the frame

    # Annotate the frame with the detected objects
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels

    # Check if annotated_frame is valid
    if annotated_frame is not None:
        # Display the annotated frame with detections in a window
        cv2.imshow("YOLOv8 Live Object Detection", annotated_frame)
    else:
        print("Error: Annotated frame is None")

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
