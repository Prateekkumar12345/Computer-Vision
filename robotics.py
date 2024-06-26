import torch
import cv2
import numpy as np

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_skeleton(frame, box):
    # Draw the bounding box of the person
    xmin, ymin, xmax, ymax = box.astype(int)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    # Convert the bounding box coordinates to the cropped region
    person_region = frame[ymin:ymax, xmin:xmax]

    # Convert the cropped region to grayscale
    gray = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours
    for contour in contours:
        # Approximate polygonal curves for the contours
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        # If the contour has more than 5 points, it might be a skeleton
        if len(approx) > 5:
            # Draw the skeleton within the bounding box
            cv2.drawContours(frame[ymin:ymax, xmin:xmax], [approx], 0, (0, 255, 0), 2)
    
    # Return the count of detected persons
    return 1

# Open video capture (change the argument to 0 for webcam)
cap = cv2.VideoCapture("video1.mp4")

# Get the original dimensions of the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window to display the frames
cv2.namedWindow('Object Detection and Skeleton Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection and Skeleton Detection', width, height)

# Initialize the count of detected persons
total_persons = 0
detected_persons = []

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLOv5
    yolo_results = yolo_model(frame)

    # Get bounding boxes and class labels from YOLOv5 results
    boxes = yolo_results.xyxy[0][:, :4].cpu().numpy()  # Bounding boxes
    class_indices = yolo_results.xyxy[0][:, 5].cpu().numpy().astype(int)  # Class indices

    # Map class indices to class labels
    class_labels = []
    for idx in class_indices:
        class_labels.append(yolo_results.names[idx])

    # Loop through detected objects
    for box, label in zip(boxes, class_labels):
        # Check if the detected object is a person
        if label == 'person':
            # Detect skeletons within the bounding box
            person_count = detect_skeleton(frame, box)
            
            # Check if the person is already in the detected_persons list
            if box.tolist() not in detected_persons:
                # If not, add the person to the list and increment the total_persons count
                detected_persons.append(box.tolist())
                total_persons += 1
    
    # Check if any detected persons are no longer in the current frame
    for person in detected_persons[:]:
        if person not in boxes.tolist():
            # If the person is no longer detected, remove them from the list and decrement the total_persons count
            detected_persons.remove(person)
            total_persons -= 1

    # Display the total number of detected persons on the frame
    cv2.putText(frame, f'Total persons: {total_persons}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detected objects and skeletons
    cv2.imshow('Object Detection and Skeleton Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()