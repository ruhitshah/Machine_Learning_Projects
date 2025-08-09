# 🎯 YOLOv8 Live Object Detection with Webcam

This Python project uses **YOLOv8** and **OpenCV** to perform **real-time object detection** from your webcam.  
It detects and labels objects live using a pre-trained YOLOv8 model.

---

## 📌 Features
- **Real-time object detection** using YOLOv8
- **Bounding boxes & labels** drawn directly on video frames
- **Configurable model size** (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, etc.)
- **Webcam live stream** with exit option

---

## 📂 Project Structure
YOLOv8-Webcam/
│
├── main.py # Python script for live object detection
├── README.md # Project documentation

---

## 🛠 Requirements

- **Python** 3.8+
- **Ultralytics YOLO** (YOLOv8)
- **OpenCV** for Python
- **Webcam** (internal or external)

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolov8-webcam.git
   cd yolov8-webcam
2. Install dependencies
pip install ultralytics opencv-python
3.Download YOLOv8 model weights

By default, yolov8n.pt is automatically downloaded from Ultralytics when you run the script.

You can also manually download models from:
Ultralytics YOLOv8 Models
python main.py
model = YOLO('yolov8n.pt')  # nano model (fast, less accurate)
⚙️ Changing the Model
# Examples:
# model = YOLO('yolov8s.pt')  # small model
# model = YOLO('yolov8m.pt')  # medium model
📜 License
This project is free to use for educational and research purposes.
✏️ Author
Developed by Ruhit Shah
Powered by Ultralytics YOLOv8 and OpenCV
