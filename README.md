**Vehicle Tracking and Counting Using Yolo V8, Sort, and Deep Sort**

**Introduction:**
This document provides an overview of a vehicle tracking project utilizing YOLOv8 (You Only Look Once version 8) for object detection and multiple tracking algorithms including SORT (Simple Online and Realtime Tracking) and DeepSORT (Deep Simple Online and Realtime Tracking).

**Objective:**
The main goal of this project is to develop a robust system for real-time vehicle tracking in video streams or recorded footage. By leveraging deep learning-based object detection and tracking algorithms, the system aims to accurately identify and track vehicles in various scenarios.

**Dependencies:**
- Python 3.x
- OpenCV
- TensorFlow or PyTorch (for YOLOv8)
- NumPy
- SORT algorithm implementation (e.g., https://github.com/abewley/sort)
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
- **Tracking (SORT and DeepSORT):** After detection, SORT or DeepSORT algorithm is employed to assign unique IDs to each detected vehicle and track them across frames.
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

**Contributing:**
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

**License:**
This project is licensed under the [MIT License](LICENSE).

**Acknowledgments:**
- YOLOv8: Citation to the original YOLOv8 implementation.
- SORT: Citation to the SORT tracker implementation.
- DeepSORT: Citation to the DeepSORT tracker implementation.

**References:**
- YOLOv8 Paper: [Link to YOLOv8 paper]
- SORT Paper: [Link to SORT paper]
- DeepSORT Paper: [Link to DeepSORT paper]
