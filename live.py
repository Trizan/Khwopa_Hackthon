from ultralytics import YOLO
import cv2
from prediction import predict
import mysql.connector

camera_id = 8

db_connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="anpr"
)

cursor = db_connection.cursor(buffered=False)

license_plate_detector = YOLO('./models/rotate_best.pt')

cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret and frame_nmr % 15 == 0:
        try:
            height, width = frame.shape[:2]
            # frame = cv2.resize(frame, (1000, int(1000 * (height / width))))

            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                
                predicted_plate, wrap_plate = predict(license_plate_crop)
                print(predicted_plate)
                
                query = f'SELECT bolo.* FROM bolo INNER JOIN records on bolo.record_id = records.id WHERE records.plate_no = "{predicted_plate}"'
                cursor.execute(query)
                row = cursor.fetchall()
                
                if row:
                    print("Record found")
                    check_query = f'SELECT id FROM hitlogs WHERE record_id={row[0][0]} AND camera_id={camera_id}'
                    cursor.execute(check_query)
                    check_row = cursor.fetchall()
                    if check_row:
                        print("Record already exists")
                        continue

                    query = f'INSERT INTO hitlogs (record_id, camera_id) VALUES ({row[0][0]},{camera_id})'
                    img_with_plate = frame.copy()
                    cv2.rectangle(img_with_plate, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    try:
                        cursor.execute(query)
                        hitlog_id = cursor.lastrowid
                        db_connection.commit()
                        cv2.imwrite(f"D:/XAMPP_7.3/htdocs/at/Snapshots/{hitlog_id}_frame.jpg", img_with_plate)
                        cv2.imwrite(f"D:/XAMPP_7.3/htdocs/at/Snapshots/{hitlog_id}_plate.jpg", wrap_plate)

                    except:
                        print("Error! looks like the record already exists")

            db_connection.commit()

        except Exception as e:
            print(e)                