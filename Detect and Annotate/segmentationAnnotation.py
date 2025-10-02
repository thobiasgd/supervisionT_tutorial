import cv2
import supervision as sv
from ultralytics import YOLO

path = 'data/03.jpg'

model = YOLO("models\\yolov8n-seg.pt")
image = cv2.imread(path)
customLabelImage = image.copy()
results = model(image)[0]

detections = sv.Detections.from_ultralytics(results)

mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

annotated_image = mask_annotator.annotate(
    scene = image, detections = detections
)
annotated_image = label_annotator.annotate(
    scene = annotated_image, detections = detections
)

cv2.imshow("Annotations", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()