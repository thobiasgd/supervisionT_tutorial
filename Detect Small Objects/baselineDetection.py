import cv2
'''
Modifying the input resolution of images before detection can enhance small object identification 
at the cost of processing speed and increased memory usage.
This method is less effective for ultra-high-resolution images (4K and above).
'''

from ultralytics import YOLO
import supervision as sv

sourcePath = 'data/beach.mov'
modelPath = 'models/yolov8x.pt'

model = YOLO(modelPath)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(sourcePath)
if not cap.isOpened():
    raise Exception("Erro ao abir a fonte de video!")

while True:
    ret, frame = cap.read()
    results = model(frame, imgsz=1504)[0]

    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = box_annotator.annotate(
        scene=frame, detections=detections
    )

    cv2.imshow('Beach', annotated_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
