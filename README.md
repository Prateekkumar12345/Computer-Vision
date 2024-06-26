Object Detection and Skeleton Detection using YOLOv5 and OpenCV
This project utilizes YOLOv5 for object detection and OpenCV for skeleton detection in a video stream. The script captures video frames, detects persons using the YOLOv5 model, and then attempts to detect skeletons within the detected person regions.

Prerequisites
Ensure you have the following libraries installed:

torch
opencv-python
numpy


The detect_skeleton function takes a frame and a bounding box as inputs, draws the bounding box, converts the region of interest to grayscale, applies binary thresholding, finds contours, and draws skeletons.

This script performs real-time object detection and skeleton detection on video frames. It uses YOLOv5 for detecting persons and OpenCV for detecting skeletons within the detected person regions. The results are displayed in a window with the total count of detected persons.
