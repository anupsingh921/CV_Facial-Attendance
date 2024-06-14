import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Directory containing student images
faces_dir = "faces/"

# Load Known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load each face image
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        try:
            face_encoding = face_recognition.face_encodings(image)[0]
        except IndexError:
            print(f"Warning: No face found in {filename}. Skipping this file.")
            continue
        known_face_encodings.append(face_encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# List of expected students
students = known_face_names.copy()

# Get the current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Open CSV file for writing attendance
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Add the text if a person is present
            if name in known_face_names:
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left * 4, top * 4), (right * 4, bottom * 4), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name + " - Present", (left * 4 + 6, bottom * 4 - 6), font, 0.5, (255, 255, 255), 1)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
