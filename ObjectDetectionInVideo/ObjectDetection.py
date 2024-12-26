pip install pytube opencv-python opencv-python-headless opencv-contrib-python tensorflow numpy
import cv2
import numpy as np
import os
from google.colab import files

# Step 1: Download YOLO Files
def download_yolo_files():
    if not os.path.exists("yolov3.weights"):
        !wget https://pjreddie.com/media/files/yolov3.weights
    if not os.path.exists("yolov3.cfg"):
        !wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    if not os.path.exists("coco.names"):
        !wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Step 2: Load YOLO Model
def load_yolo_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    
    # Handle different formats of getUnconnectedOutLayers() output
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers, np.ndarray):  # If ndarray, flatten it
        unconnected_layers = unconnected_layers.flatten()
    
    try:
        output_layers = [layer_names[int(i) - 1] for i in unconnected_layers]
    except (TypeError, IndexError):
        output_layers = [layer_names[int(i[0]) - 1] for i in unconnected_layers]

    return net, classes, output_layers

# Step 3: Process Video Frames
def detect_objects(frame, net, output_layers, classes):
    height, width, channels = frame.shape
    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # Process detections
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

# Step 4: Main Function for Object Detection on Video
def process_video(video_path):
    # Load YOLO model
    net, classes, output_layers = load_yolo_model()

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_objects(frame, net, output_layers, classes)
        out.write(frame)
    
    cap.release()
    out.release()
    print("Processing complete. Output saved as 'output.mp4'.")

# Step 5: File Upload and Execution
if __name__ == "__main__":
    # Download YOLO files
    download_yolo_files()

    # Upload the video file
    print("Please upload your video file:")
    uploaded = files.upload()
    video_path = next(iter(uploaded))  # Get the uploaded file name

    # Process the video
    process_video(video_path)

    # Download the output video
    print("Downloading output video...")
    files.download("output.mp4")
