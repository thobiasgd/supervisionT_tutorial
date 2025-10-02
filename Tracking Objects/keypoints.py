import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

sourcePath = 'data/skiing.mp4'
modelPath = 'models/yolov8m-pose.pt'

model = YOLO(modelPath)

edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    key_points = sv.KeyPoints.from_ultralytics(results)

    annotated_frame = edge_annotator.annotate(
        frame.copy(), key_points=key_points
    )
    return vertex_annotator.annotate(
        annotated_frame, key_points=key_points
    )

sv.process_video(
    source_path=sourcePath,
    target_path= 'data/keypointsResults.mp4',
    callback=callback
)