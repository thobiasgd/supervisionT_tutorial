import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

sourcePath = 'data/skiing.mp4'
modelPath = 'models/yolov8m-pose.pt'

model = YOLO(modelPath)

edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()
box_annotator = sv.BoxAnnotator()
trace_annotator = sv.TraceAnnotator()

tracker = sv.ByteTrack()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)
    detections = key_points.as_detections()
    detections = tracker.update_with_detections(detections)

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points
    )
    vertex_annotator.annotate(
        annotated_frame, key_points=key_points
    )
    box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    return trace_annotator.annotate(
        annotated_frame, detections
    )

sv.process_video(
    source_path=sourcePath,
    target_path= 'data/keypointsAndTrackingResults.mp4',
    callback=callback
)