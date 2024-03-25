# Vehicle Tracking and Counting Using Yolo V8 and DeepSORT

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

In this project, we will perform vehicle tracking and counting using YOLOv8 and DeepSORT (Deep Simple Online and Realtime Tracking).

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training](#training)
   - [Data Preparation](#data-preparation)
   - [Model Configuration](#model-configuration)
   - [Training Process](#training-process)
5. [Inference](#inference)
6. [Performance Tuning](#performance-tuning)

## 1. Introduction

The ability to perform both semantic segmentation and depth estimation in real-time has applications in autonomous driving, robotics, augmented reality, and more. This project aims to provide a practical and accessible implementation of a multi-task network capable of handling these tasks simultaneously.

### 1.1. A brief explanation of the Model

YOLOv8 (You Only Look Once version 8) is an object detection algorithm that operates by dividing the input image into a grid and predicting bounding boxes and class probabilities directly from a single neural network evaluation. YOLOv8 improves upon its predecessors by incorporating advancements such as a feature pyramid network, anchor box clustering, and advanced training techniques like multi-scale training and data augmentation. It leverages a deep convolutional architecture, typically based on Darknet or similar frameworks, to efficiently process images and make predictions in real-time. YOLOv8 excels in various computer vision tasks, including object detection in images and videos, due to its balance between accuracy, speed, and simplicity, making it widely utilized in applications ranging from autonomous vehicles to surveillance systems.

DeepSORT (Deep Learning-based SORT) is a tracking algorithm that combines the Simple Online and Realtime Tracking (SORT) algorithm with deep learning techniques to achieve robust and accurate object tracking in videos. It integrates a deep association metric to improve the matching process between detections in consecutive frames, enabling it to handle occlusions, clutter, and appearance changes more effectively. DeepSORT utilizes a convolutional neural network (CNN) to generate embeddings representing the appearance of detected objects, which are then used to compute similarity scores for association. By incorporating both motion and appearance cues, DeepSORT achieves state-of-the-art performance in multi-object tracking tasks, making it a widely adopted solution in various real-world applications such as surveillance, autonomous driving, and sports analytics.


## 2. Getting Started

**Dependencies:**
- Python 3.x
- OpenCV
- PyTorch (for YOLOv8)
- NumPy
- DeepSORT algorithm implementation (e.g., https://github.com/nwojke/deep_sort)

**Installation:**
1. Install Python 3.x if not already installed.
2. Install required Python packages using pip:
   ```
   pip install opencv-python tensorflow numpy
   ```
3. Clone SORT and DeepSORT repositories and follow their respective installation instructions.

**Usage:**
1. Obtain the pre-trained YOLOv8 model weights and configuration file.
2. Set up the SORT and DeepSORT trackers according to their documentation.
3. Integrate YOLOv8 for vehicle detection within your tracking script.
4. Define input video streams or paths to recorded footage.
5. Implement the tracking pipeline, including detection, tracking, and visualization.
6. Execute the tracking script.

**Functionality:**
- **Object Detection (YOLOv8):** YOLOv8 is used to detect vehicles in each frame of the input video stream.
- **Tracking (SORT and DeepSORT):** After detection, DeepSORT algorithm is employed to assign unique IDs to each detected vehicle and track them across frames.
- **Visualization:** The tracked vehicles can be visualized using bounding boxes and unique IDs on the video frames.

**Example Code:**
```python
import cv2
from sort import Sort
from deep_sort import DeepSort
from yolo_v8 import YOLOv8

# Initialize YOLOv8 for vehicle detection
yolo = YOLOv8()

# Initialize SORT tracker
sort_tracker = Sort()

# Initialize DeepSORT tracker
deepsort_tracker = DeepSort()

# Capture video stream
cap = cv2.VideoCapture('input_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform vehicle detection using YOLOv8
    detections = yolo.detect_vehicles(frame)
    
    # Track vehicles using SORT
    sorted_tracks = sort_tracker.update(detections)
    
    # Track vehicles using DeepSORT
    deep_sorted_tracks = deepsort_tracker.update(detections)
    
    # Visualize tracked vehicles
    # (code to draw bounding boxes and IDs on frame)
    
    # Display the frame
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## References
- YOLOv8 Paper: [Link to YOLOv8 paper]
- DeepSORT Paper: [Link to DeepSORT paper]
