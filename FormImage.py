from ultralytics import YOLO
import cv2
from prediction import predict
import numpy as np


def get_plate_characters(frame):
    predicted_plate = ''

    license_plate_detector = YOLO('./models/rotate_best.pt')

    results = license_plate_detector(frame, save=True, project="E:/Khwopa_Hackathon/Read_license_plate/test")

    license_plates = results[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = license_plate
        if confidence > 0.4:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            results = predict(license_plate_crop)
            if results is not None:
                predicted_plate, wrap_plate = predict(license_plate_crop)
    print(predicted_plate)
    return predicted_plate
