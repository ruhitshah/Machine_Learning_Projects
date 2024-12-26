pip install opencv-python
# Step 1: Download the necessary YOLO files
!wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
!wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg -O yolov3.cfg
!wget https://github.com/pjreddie/darknet/raw/master/data/coco.names -O coco.names

# Step 2: Check if the files have been downloaded
!ls

# Step 3: Import required libraries
import cv2
import numpy as np
from google.colab import files
import os

# Step 4: Upload the image for object detection
uploaded = files.upload()

# Step 5: Check if the YOLO files exist
if os.path.exists("yolov3.weights") and os.path.exists("yolov3.cfg") and os.path.exists("coco.names"):
    # Step 6: Load YOLO pre-trained weights and configuration
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Step 7: Load the COCO names file (the objects the model can detect)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Step 8: Load the selected image
    image_path = list(uploaded.keys())[0]
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # Step 9: Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Step 10: Process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Only keep detections with confidence > 0.5
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Step 11: Perform Non-Maximum Suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Step 12: Draw the bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw the rectangle around detected objects
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the label
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Step 13: Save the output image
    output_image_path = "/content/detected_image.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Detected image saved at: {output_image_path}")

    # Step 14: Display the output image using matplotlib (for Google Colab)
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.title("Detected Objects")
    plt.show()

else:
    print("YOLO files (weights, cfg, coco.names) are missing. Please check your downloads.")
