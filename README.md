# Vehicle Counting and Tracking using YOLOv8 and ByteTrack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 1. Introduction

This project aims to create a vehicle tracking and counting system using YOLOv8 and ByteTRACK. It offers a reliable and efficient system for analyzing traffic flow, monitoring congestion, and enhancing overall road safety. 

## 1.1. Features

- **YOLOv8**: YOLOv8 is utilized for accurate and real-time vehicle detection.
  
- **ByteTrack**: ByteTrack algorithm is employed for multi-object tracking, ensuring smooth and reliable tracking of vehicles across frames.

- **Line Counter**: The system efficiently counts the number of vehicles passing through a line, aiding in traffic analysis and management.

## 2. Usage
### 2.1. Dependencies

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- etc. (list any other dependencies here)

### 2.2. Installation

1. Clone the repository:

   ```
   git clone https://github.com/username/repository.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Download pretrained weights for YOLOv8 and ByteTrack (provide links or instructions on how to obtain these weights).

### 2.3. Usage

1. Configure input source: Modify the configuration file to specify the input source (e.g., camera feed, video file).

2. Run detection and tracking:

   ```
   python main.py
   ```

3. View the output: The system will generate visualizations displaying vehicle detection and tracking results.


## 3. Acknowledgements

This project is heavily based on tutorial by [Roboflow](https://github.com/roboflow) in this [colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb#scrollTo=Q9ppb7bFvWfc). It works by combining the YOLOv8 model from Ultralytics and ByteTRACK model developed by Yifu Zhange, et al. The links to the YOLOv8 and and ByteTRACK repository are:

- YOLOv8: [Link to YOLOv8 repository](https://github.com/ultralytics/ultralytics)
- ByteTrack: [Link to ByteTrack repository](https://github.com/ifzhang/ByteTrack)
