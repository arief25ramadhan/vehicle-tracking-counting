import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
       
        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8m model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def get_results(self, results):
        
        xyxys = []
        confidences = []
        class_ids = []
        detections_list = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            #if class_id == 0:
                
            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            
            merged_detection = ([bbox[0][0], bbox[0][1], bbox[0][2]-bbox[0][0], bbox[0][3]-bbox[0][1]], confidence, class_id)
            
            detections_list.append(merged_detection)
            xyxys.append(bbox)
            confidences.append(confidence)
            class_ids.append(class_id)
            
    
        return detections_list
    
    
    def draw_bounding_boxes(self, img, bboxes, ids):
  
        for bbox, id_ in zip(bboxes, ids):
        
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #cv2.putText(img, "Conf: " + str(int(score * 100)) + "%", (int(bbox[2]-200), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return img
    
    
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        
        # SORT
        sort = Sort(max_age=30, min_hits=20, iou_threshold=0.5)
        
        # DEEPSORT
        tracker = DeepSort(max_age=5, n_init=10)
            
        
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            detections_list = self.get_results(results)
            
        
            
            tracks = tracker.update_tracks(detections_list, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            
            
            #  Get the bboxes from the tracks in the tracker
            detections_list = []
            ids = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                ids.append(track.track_id)
                detections_list.append(track.to_ltrb().tolist())
                
                track_id = track.track_id
                bbox = track.to_ltrb()
                
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
                cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture_index=2)
detector()
