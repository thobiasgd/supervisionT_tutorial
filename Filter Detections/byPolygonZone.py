import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np

sourcePath = 'data/03.jpg'
modelPath = YOLO("models/yolov8n.pt")

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

polygon = np.array([[434, 13], 
                    [242, 385],  
                    [590, 386], 
                    [590, 20]])

zone = sv.PolygonZone(polygon=polygon)

results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)
mask = zone.trigger(detections=detections)
detections = detections[mask]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
zone_annotator = sv.PolygonZoneAnnotator(zone = zone, color = sv.Color.RED)

annotated_image = box_annotator.annotate(
    scene=img, detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections
)
annotated_image = zone_annotator.annotate(
    scene=annotated_image
)


cv2.imshow("Cars Only", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()