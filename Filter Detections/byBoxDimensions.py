import supervision as sv
import cv2
from ultralytics import YOLO

sourcePath = 'data/01.jpeg'
modelPath = YOLO("models/yolov8n.pt")

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)

width = detections.xyxy[:, 2] - detections.xyxy[:, 0]
height = detections.xyxy[:, 3] - detections.xyxy[:, 1]

detections = detections[(width > 100) & (height > 100)]

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