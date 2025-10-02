import cv2
from ultralytics import YOLO
import supervision as sv

videoPath = 'data/traffic.mp4'
model = YOLO("models/yolov8n.pt")

frames_generator = sv.get_video_frames_generator(videoPath)

with sv.CSVSink("Saved Files/csvFile.csv") as sink:
    for frame in frames_generator:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        sink.append(detections, {})

