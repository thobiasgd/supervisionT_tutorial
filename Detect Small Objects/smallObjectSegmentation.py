'''
InferenceSlicer can perform segmentation tasks too.
'''

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

sourcePath = 'data/beach.jpg'
modelPath = 'models/yolov8n-seg.pt'

img = cv2.imread(sourcePath)
model = YOLO(modelPath)

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback=callback)
detections = slicer(img)

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(thickness = 5)

annotated_image = mask_annotator.annotate(
    scene=img, detections=detections)
annotated_img = box_annotator.annotate(
    scene=annotated_image, detections=detections
)

cv2.imshow('Beach', cv2.resize(annotated_img, (img.shape[1] // 5, img.shape[0] // 5)))

cv2.waitKey(0)
cv2.destroyAllWindows()
