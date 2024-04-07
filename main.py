# Import Libraries
import os
import sys
import numpy as np
from utils import *
from tqdm.notebook import tqdm
from ultralytics import YOLO
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from yolox.tracker.byte_tracker import BYTETracker, STrack


class vehicle_tracker_and_counter:

    def __init__(self,
                source_video_path="assets/vehicle-counting.mp4",
                target_video_path="assets/vehicle-counting-result.mp4"):

        # YOLOv8 Object Detector
        self.model_name = "yolov8x.pt"
        self.model = YOLO(model_name)
        self.model.fuse()
  
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
        self.generator = get_video_frames_generator(self.SOURCE_VIDEO_PATH)
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
                results = self.model(frame)
                detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                )
                # filtering out detections with unwanted classes
                mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                detections.filter(mask=mask, inplace=True)
                # tracking detections
                tracks = byte_tracker.update(
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
                    f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
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
    pipeline = vehicle_tracker_and_counter(source_video_path=input_video, target_video_path=output_video)
    pipeline.run()
