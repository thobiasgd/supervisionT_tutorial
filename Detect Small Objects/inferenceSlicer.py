'''
InferenceSlicer processes high-resolution images by dividing them into smaller segments, 
detecting objects within each, and aggregating the results.
'''

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

sourcePath = 'data/beach.jpg'
modelPath = 'models/yolov8n.pt'

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback=callback)
detections = slicer(img)

box_annotator = sv.BoxAnnotator(thickness = 5)

annotated_img = box_annotator.annotate(
    scene=img, detections=detections
)

cv2.imshow('Beach', cv2.resize(annotated_img, (img.shape[1] // 5, img.shape[0] // 5)))

cv2.waitKey(0)
cv2.destroyAllWindows()
