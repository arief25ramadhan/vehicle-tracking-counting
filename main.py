# Import Libraries
import os
import cv2
import sys
import time
import numpy as np
from utils import *
from tqdm import tqdm
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack


class vehicle_tracker_and_counter:

    def __init__(self,
                source_video_path="assets/vehicle-counting.mp4",
                target_video_path="assets/vehicle-counting-result.mp4",
                use_tensorrt=False):
        
        # YOLOv8 Object Detector
        self.model_name = "yolov8x.pt"
        self.yolo = YOLO(self.model_name)

        if use_tensorrt:
            try: 
                # Try to load model if it is already exported
                self.model = YOLO('yolov8x.engine')
            except:
                # Export model
                self.yolo.export(format='engine')  # creates 'yolov8x.engine'
                # Load the exported TensorRT model
                self.model = YOLO('yolov8x.engine')
        else:
            self.model = self.yolo
            self.model.fuse()

        self.CLASS_NAMES_DICT = self.yolo.model.names
        self.CLASS_ID = [2, 3, 5, 7]
  
        # Line for counter
        self.line_start = Point(50, 1500)
        self.line_end = Point(3840-50, 1500)

        # BYTETracke Object Tracker
        self.byte_tracker = BYTETracker(BYTETrackerArgs())

        # Video input and output path
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        # Create VideoInfo instance
        self.video_info = VideoInfo.from_video_path(self.source_video_path)
        # Create frame generator
        self.generator = get_video_frames_generator(self.source_video_path)
        # Create LineCounter instance
        self.line_counter = LineCounter(start=self.line_start, end=self.line_end)
        # Create instance of BoxAnnotator and LineCounterAnnotator
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
        self.line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)
            

    def run(self):
        # Open target video file
        with VideoSink(self.target_video_path, self.video_info) as sink:
            # loop over video frames
            for frame in tqdm(self.generator, total=self.video_info.total_frames):
                # model prediction on single frame and conversion to supervision Detections
                start_time = time.time()
                results = self.model(frame)
                end_time = time.time()
                fps = np.round(1/(end_time - start_time), 2)
                cv2.putText(frame, f'FPS: {fps}s', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)

                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # filtering out detections with unwanted classes
                mask = np.array([class_id in self.CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # tracking detections
                tracks = self.byte_tracker.update(
                    output_results=detections2boxes(detections=detections),
                    img_info=frame.shape,
                    img_size=frame.shape
                )
                tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                detections.tracker_id = np.array(tracker_id)
                # filtering out detections without trackers
                mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # format custom labels
                labels = [
                    f"#{tracker_id} {self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]
                # updating line counter
                self.line_counter.update(detections=detections)
                # annotate and display frame
                frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=labels)
                self.line_annotator.annotate(frame=frame, line_counter=self.line_counter)
                sink.write_frame(frame)

if __name__ == '__main__':

    input_video="assets/vehicle-counting.mp4"
    output_video="assets/vehicle-counting-result.mp4"
    pipeline = vehicle_tracker_and_counter(source_video_path=input_video, target_video_path=output_video, use_tensorrt=True)
    pipeline.run()
