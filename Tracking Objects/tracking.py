import numpy as np
from ultralytics import YOLO
import supervision as sv
import cv2

modelPath = 'models/yolov8m.pt'
sourcePath = 'data/people-walking.mp4'

model = YOLO(modelPath) 
tracker = sv.ByteTrack() # Rastrador do tipo movimento
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {class_name}"
        for class_name, tracker_id
        in zip(detections.data["class_name"], detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )
    return label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

sv.process_video(
    source_path=sourcePath,
    target_path='data/trackingResults.mp4',
    callback=callback
)

