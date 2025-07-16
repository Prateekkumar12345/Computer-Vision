# 🧍‍♂️ YOLOv5 Person Detection & Skeleton Contour Approximation

This project performs **real-time object detection** using YOLOv5 to detect **persons in video frames**, and applies basic **contour-based skeleton approximation** within each detected bounding box using OpenCV.

---

## 📌 Features

- 🧠 **YOLOv5** (`yolov5s`) pre-trained model from Ultralytics via PyTorch Hub
- 🎥 Real-time or file-based video processing
- 🔲 Person detection via bounding boxes
- 🧬 Skeleton-like contour approximation using OpenCV on detected persons
- 🧾 Dynamic tracking of total detected persons
- 💻 Live visualization with OpenCV GUI

---

## 📦 Dependencies

Install the required packages:

```bash
pip install torch torchvision opencv-python numpy

Project Structure
├── detect_skeleton.py       # Main script
├── video1.mp4               # Sample video (replaceable)
├── README.md                # This documentation
