# Vehicle Counting and Tracking using YOLOv8 and ByteTrack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 1. Introduction

This project aims to create a vehicle tracking and counting system using the YOLOv8 and ByteTrack models. It offers a reliable and efficient system for analyzing traffic flow, monitoring congestion, and enhancing overall road safety. The main components of the project are:
- **YOLOv8**: The YOLOv8 model from Ultralytics is utilized for accurate and real-time vehicle detection.  
- **ByteTrack**: ByteTrack algorithm is employed for multi-object tracking, ensuring smooth and reliable tracking of vehicles across frames.
- **Line Counter**: We use the supervision library to count the number of vehicles entering or leaving a region.

## 2. Installation

To use this repository, we need to set up our environment with its required libraries. The steps are:

1. Clone the repository:

   ```
   git clone https://github.com/arief25ramadhan/YOLOv8-Vehicle-Tracking-Counting.git
   ```

2. Go to the repository, and install dependencies:

   ```
   cd YOLOv8-Vehicle-Tracking-Counting
   pip install -r requirements.txt
   ```

3. Inside this current repo, clone the ByteTrack Libraries

    ```
   git clone https://github.com/ifzhang/ByteTrack.git
   ```
    
5. Install ByteTrack dependencies
   ```
   cd ByteTrack
   pip install -r requirements.txt
   ```


## 3. Usage

1. Configure input source: Modify the configuration file to specify the input source (e.g., camera feed, video file).

2. Run detection and tracking:

   ```
   python main.py
   ```

3. View the output: The system will generate visualizations displaying vehicle detection and tracking results.


## 4. Results



## References

This project is heavily based on tutorial by [Roboflow](https://github.com/roboflow) in this [colab notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-track-and-count-vehicles-with-yolov8.ipynb#scrollTo=Q9ppb7bFvWfc). It works by combining the YOLOv8 model from Ultralytics and ByteTrack model developed by Yifu Zhange, et al. The links to the YOLOv8 and and ByteTrack repository are:

- YOLOv8: [Link to YOLOv8 repository](https://github.com/ultralytics/ultralytics)
- ByteTrack: [Link to ByteTrack repository](https://github.com/ifzhang/ByteTrack)
