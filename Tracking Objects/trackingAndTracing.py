import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

sourcePath = 'data/people-walking.mp4'
modelPath = 'models/yolov8m.pt'

model = YOLO(modelPath)
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {class_name}"
        for tracker_id, class_name
        in zip(detections.data['class_name'], detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )
    label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels = labels
        )

    return trace_annotator.annotate(annotated_frame, detections)

sv.process_video(
    source_path=sourcePath,
    target_path='data/trackingAndTracingResults.mp4',
    callback=callback
)