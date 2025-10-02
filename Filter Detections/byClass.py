import supervision as sv
import cv2
from ultralytics import YOLO

sourcePath = 'data/04.png'
modelPath = YOLO("models/yolov8n.pt")

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.class_id == 0]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=img, detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections
)

cv2.imshow("Cars Only", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()