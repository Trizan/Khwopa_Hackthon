import cv2
import numpy as np
from ultralytics import YOLO

def found_vehicle(frame, model_path="E:/Khwopa_Hackathon/Read_license_plate/models/yolov8n.pt", target_size=(640, 640)):
    """
    Detect vehicles in a frame, crop and resize each vehicle image
    
    Args:
        frame (numpy.ndarray): Input image frame
        model_path (str): Path to YOLO model
        target_size (tuple): Desired size for cropped images
    
    Returns:
        list: Cropped and resized vehicle images
    """
    # Load YOLO model
    model = YOLO(model_path)
    
    # Detect vehicles
    results = model(frame, save=True, project="E:/Khwopa_Hackathon/Read_license_plate/test")
    
    # List to store cropped vehicle images
    cropped_vehicles = []
    
    # Process each detected vehicle
    for result in results:
        boxes = result.boxes
        
        for box in boxes.data.tolist():
            # Get bounding box coordinates
            x1, y1, x2, y2, confidence, class_id = box
            
            if confidence > 0.2:
                # Convert coordinates to integers for slicing
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure indices are within image boundaries
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                
                # Crop vehicle image
                cropped_vehicle = frame[y1:y2, x1:x2]

                # Resize/stretch to target size
                resized_vehicle = cv2.resize(cropped_vehicle, target_size, interpolation=cv2.INTER_AREA)

                cropped_vehicles.append(resized_vehicle)
    
    return cropped_vehicles

#print(found_vehicle(cv2.imread("E:/Khwopa_Hackathon/Read_license_plate/nepal-rental-car.jpg")))
if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("E:/Khwopa_Hackathon/Read_license_plate/models/detect_vehicle.pt")
    
    # Detect vehicles
    results = model.predict("E:/Khwopa_Hackathon/Read_license_plate/nepal-rental-car.jpg", save=True, project="E:/Khwopa_Hackathon/Read_license_plate/test")
