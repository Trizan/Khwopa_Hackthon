import os
import django

# Set the environment variable for settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Admin_panel.settings')  # Replace with your settings module

# Initialize Django
django.setup()


import csv
from django.db import IntegrityError
from surveliance.models import VehicleLog, VehicleContact  # Adjust imports
from twilio.rest import Client  # Twilio SDK
import time
from datetime import datetime

from ultralytics import YOLO
import cv2
import numpy as np
from prediction import predict  # Assuming `predict` extracts characters from license plates
from FindVehicle import found_vehicle
from FormImage import get_plate_characters

class Vehicle:
    def __init__(self, license_plate, counter=1):
        self.license_plate = license_plate
        self.counter = counter
        self.last_seen = datetime.now()

    def increment_counter(self):
        self.counter += 1
        self.last_seen = datetime.now()

    def to_csv_row(self):
        return [self.license_plate, self.counter, self.last_seen.strftime('%Y-%m-%d %H:%M:%S')]


class TwilioHandler:
    def __init__(self, account_sid, auth_token, from_phone):
        self.client = Client(account_sid, auth_token)
        self.from_phone = from_phone

    def send_sms(self, to_phone, message):
        """Send SMS using Twilio."""
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=to_phone
            )
            print(f"SMS sent successfully: SID {message.sid}")
        except Exception as e:
            print(f"Failed to send SMS: {e}")


class VehicleTracker:

    def __init__(self, log_file_path, sms_handler):
        self.log_file_path = log_file_path
        self.vehicle_objects = {}  # Tracks Vehicle objects by license_plate
        self.sms_handler = sms_handler  # Twilio handler instance

    def load_existing_records(self):
        """Load existing records from the log file into vehicle objects."""
        try:
            with open(self.log_file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    license_plate, counter, last_seen = row
                    self.vehicle_objects[license_plate] = Vehicle(
                        license_plate=license_plate,
                        counter=int(counter),
                    )
        except FileNotFoundError:
            pass  # Start fresh if no log file exists

    def save_records(self):
        """Save all vehicle objects to the log file."""
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            print("Here")
            writer.writerow(['license_plate', 'counter', 'last_seen'])  # Header
            for vehicle in self.vehicle_objects.values():
                print(vehicle)
                writer.writerow(vehicle.to_csv_row())

    def get_contact_info(self, license_plate):
        """Fetch contact information from the database."""
        try:
            vehicle_contact = VehicleContact.objects.get(Vehicle_Number=license_plate)
            return vehicle_contact.contact
        except VehicleContact.DoesNotExist:
            print(f"No contact info found for vehicle {license_plate}")
            return None

    def send_alert(self, license_plate):
        """Send an alert for vehicles violating the rules."""
        contact_info = self.get_contact_info(license_plate)
        if contact_info:
            message = f"Alert: Vehicle {license_plate} has reached counter 2! Please check."
            self.sms_handler.send_sms(contact_info, message)
            print(f"Alert sent to {contact_info} for vehicle {license_plate}")
        else:
            print(f"Could not send alert for vehicle {license_plate} - No contact info")

    def update_django_database(self, license_plate):
        """Update the Django database with fine details."""
        try:
            # Update or create record in Django database
            vehicle, created = VehicleLog.objects.update_or_create(
                license_plate=license_plate,
                defaults={
                    'counter': self.vehicle_objects[license_plate].counter,
                    'last_seen': datetime.now(),
                }
            )
            if not created:
                vehicle.counter += 1
                vehicle.save(update_fields=['counter'])

            vehicle.charges = 500 * vehicle.counter  # Increment fine per violation
            vehicle.save(update_fields=['charges'])

            print(f"Database updated for {license_plate} - Fine: {vehicle.charges}")
        except IntegrityError as e:
            print(f"Database error: {e}")

    def update_log_and_track_violations(self, found_vehicles):
        """Process found vehicles and update log."""
        seen_this_batch = set()

        for vehicle_img in found_vehicles:
            lc_plate_number = get_plate_characters(vehicle_img)
            if lc_plate_number == '':
                continue

            seen_this_batch.add(lc_plate_number)

            if lc_plate_number in self.vehicle_objects:
                vehicle = self.vehicle_objects[lc_plate_number]
                vehicle.increment_counter()
            else:
                self.vehicle_objects[lc_plate_number] = Vehicle(lc_plate_number)

            # Check for specific counter thresholds
            if self.vehicle_objects[lc_plate_number].counter >= 2:
                self.send_alert(lc_plate_number)
            elif self.vehicle_objects[lc_plate_number].counter == 5:
                self.update_django_database(lc_plate_number)

        # Remove vehicles not seen in this batch
        for plate in list(self.vehicle_objects.keys()):
            if plate not in seen_this_batch:
                del self.vehicle_objects[plate]

        # Save updated records to the log
        self.save_records()


def capture_video_frames(video_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_count = 0
    last_capture_time = time.time()
    
    try:
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            # Break loop if no more frames
            if not ret:
                break
            
            current_time = time.time()

            # Twilio credentials
            ACCOUNT_SID = "AC134df6424e614da83833f8ceb30fe1b8"
            AUTH_TOKEN = "fd412cc017004845966046bd6afbef69"
            FROM_PHONE = "+17756185496"  # Replace with your Twilio phone number

            # Initialize Twilio SMS handler
            sms_handler = TwilioHandler(account_sid=ACCOUNT_SID, auth_token=AUTH_TOKEN, from_phone=FROM_PHONE)

            # Initialize Vehicle Tracker
            tracker = VehicleTracker(log_file_path="log.csv", sms_handler=sms_handler)
            tracker.load_existing_records()
            
            # Capture frame every 20 seconds
            if current_time - last_capture_time >= 10:
                # Save frame
                cv2.imwrite(f'frame_{frame_count}.jpg', frame)
                print(f"Captured frame at {current_time}")
                found_vehicles = found_vehicle(frame) # List of vehicle images from detection process
                tracker.update_log_and_track_violations(found_vehicles)
                # Reset capture time
                last_capture_time = current_time
                frame_count += 1

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

capture_video_frames("E:/Khwopa_Hackathon/Read_license_plate/VID20241207091221.mp4")

def image_detect(image_path):
        last_capture_time = time.time()
        while True:
            
            current_time = time.time()

            # Twilio credentials
            ACCOUNT_SID = "AC134df6424e614da83833f8ceb30fe1b8"
            AUTH_TOKEN = "fd412cc017004845966046bd6afbef69"
            FROM_PHONE = "+17756185496"  # Replace with your Twilio phone number

            # Initialize Twilio SMS handler
            sms_handler = TwilioHandler(account_sid=ACCOUNT_SID, auth_token=AUTH_TOKEN, from_phone=FROM_PHONE)

            # Initialize Vehicle Tracker
            tracker = VehicleTracker(log_file_path="log.csv", sms_handler=sms_handler)
            tracker.load_existing_records()
            
            # Capture frame every 20 seconds
            if current_time - last_capture_time >= 10:
                # Save frame
                frame = cv2.imread(image_path)
                print(f"Captured frame at {current_time}")
                found_vehicles = found_vehicle(frame) # List of vehicle images from detection process
                tracker.update_log_and_track_violations(found_vehicles)
                # Reset capture time
                last_capture_time = current_time
            time.sleep(20)
    


