import cv2
from svm_predict import recognize
import numpy as np
import functools
from boundary_fill import boundary_fill_cython  


def segment_characters(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    component_sizes = [np.sum(labels == label) for label in range(1, np.max(labels) + 1)]

    largest_component_size = max(component_sizes)
    
    lower = largest_component_size * 0.1
    upper = largest_component_size + 10
    
    for (i, label) in enumerate(np.unique(labels)):
        
        if label == 0:
            continue
    
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 10:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare) )

    plate_no = ''
    
    for rect in boundingBoxes:
        x,y,w,h = rect

        crop = image[y:y+h, x:x+w]
        plate_no += recognize(crop)
        
    return plate_no


def predict(img):
    try:
        # Check if image is valid
        if img is None or img.size == 0:
            print("Error: Invalid or empty image")
            return None

        # Resize image maintaining aspect ratio
        aspect_ratio = img.shape[1] / img.shape[0]
        img = cv2.resize(img, (500, int(500 // aspect_ratio)))

        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        bfilter = cv2.bilateralFilter(img, 11, 17, 17)  
        edges = cv2.Canny(bfilter, 50, 150) 

        kernel = np.ones((2, 2), np.uint8)
        dialated_edges = cv2.dilate(edges, kernel, iterations=1)

        img_boundary = dialated_edges.copy()
        fill_color = 255
        boundary_color = 255  
        boundary_coordinates = []
        
        # Add more robust error handling for boundary fill
        try:
            _, boundary_coordinates = boundary_fill_cython(img_boundary, (center_x, center_y), fill_color, boundary_color)
        except Exception as fill_error:
            print(f"Error in boundary fill: {fill_error}")
            return None

        boundary_points = np.array(boundary_coordinates, dtype=np.int32)

        hull = cv2.convexHull(boundary_points)
        hull_area = cv2.contourArea(hull)

        # Prevent infinite loop
        max_attempts = 10
        attempt = 0
        while hull_area < 0.1 * img.shape[0] * img.shape[1] and attempt < max_attempts:
            boundary_coordinates = []
            center_x += 10
            img_boundary = dialated_edges.copy()
            
            try:
                _, boundary_coordinates = boundary_fill_cython(img_boundary, (center_x, center_y), fill_color, boundary_color)
            except Exception as fill_error:
                print(f"Error in boundary fill: {fill_error}")
                return None

            boundary_points = np.array(boundary_coordinates, dtype=np.int32)
            hull = cv2.convexHull(boundary_points)
            hull_area = cv2.contourArea(hull)
            attempt += 1

        if attempt == max_attempts:
            print("Maximum attempts reached. Could not find suitable boundary.")
            return None

        epsilon = 0.03 * cv2.arcLength(hull, True)
        approx_polygon = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx_polygon) == 4:
            corner_points = approx_polygon.reshape(-1, 2)
            
            corner_points = corner_points[np.argsort(corner_points.sum(axis=1))]
            
            region_width = (corner_points[2,0] - corner_points[0,0])
            region_height = (corner_points[1,1] - corner_points[0,1])

            aspect_ratio_region = region_width / region_height

            width = 500
            height = int(500 / aspect_ratio_region)
            
            target_rect = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

            transformation_matrix = cv2.getPerspectiveTransform(corner_points.astype(np.float32), target_rect)

            img1 = img.copy()
            result_img = cv2.warpPerspective(img1, transformation_matrix, (width, height))
            
            # Segment characters and return
            plate_number = segment_characters(result_img)
            
            # Validate plate number
            if plate_number and len(plate_number) > 0:
                return plate_number, result_img
            else:
                print("No valid plate number found")
                return None
        else:
            print("Could not find four corner points. Adjust parameters or preprocess the image.")
            return None

    except Exception as e:
        print(f"Unexpected error in predict function: {e}")
        return None
    
if __name__ == "__main__":
    img = cv2.imread("E:\\Khwopa_Hackathon\\dataset\\images\\1714911342978.jpg")
    if img is None:
        print("Error: Could not load image. Check the file path or file format.")
        exit()

    print(predict(img))
    cv2.waitKey(0)