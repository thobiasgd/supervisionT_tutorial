""" Allows you to select detections based on their size in relation to the size of whole image. 
Sometimes the concept of detection size changes depending on the image. 
Detection occupying 10000 square px can be large on a 1280x720 image but small on a 3840x2160 image. 
In such cases, we can filter out detections based on the percentage of the image area occupied by them. 
In the example below, we remove too large detections. """

import supervision as sv
import cv2
from ultralytics import YOLO

sourcePath = 'data/01.jpeg'
modelPath = YOLO("models/yolov8n.pt")

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

height, width, channels = img.shape
image_area = height * width

results = model(img)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[(detections.area / image_area) > 0.05]

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