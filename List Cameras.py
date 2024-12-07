import cv2

def test_camera_index(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return False
    cap.release()
    return True

def find_camera_indices():
    indices = []
    for index in range(10):  # Check indices from 0 to 9
        if test_camera_index(index):
            indices.append(index)
    return indices

available_camera_indices = find_camera_indices()
print("Available Camera Indices:", available_camera_indices)
