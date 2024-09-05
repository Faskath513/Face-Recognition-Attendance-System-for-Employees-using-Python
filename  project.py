import os
import cv2 # type: ignore
import face_recognition # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from datetime import datetime
import pickle
from fpdf import FPDF # type: ignore

# Step 1: Capture Images
def capture_images(employee_name):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Images")
    img_counter = 0
    os.makedirs(f"dataset/{employee_name}", exist_ok=True)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow("Capture Images", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:  # SPACE pressed
            img_name = f"dataset/{employee_name}/image_{img_counter}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

# Step 2: Encode Faces
def encode_faces():
    encoded_faces = {}
    for employee_name in os.listdir('dataset'):
        employee_folder = os.path.join('dataset', employee_name)
        if not os.path.isdir(employee_folder):
            continue
        for img_name in os.listdir(employee_folder):
            img_path = os.path.join(employee_folder, img_name)

            # Load image and align face
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image, model="cnn")  # Use CNN model for better accuracy
            encodings = face_recognition.face_encodings(image, face_locations)

            if encodings:
                encoding = encodings[0]
                if employee_name in encoded_faces:
                    encoded_faces[employee_name].append(encoding)
                else:
                    encoded_faces[employee_name] = [encoding]

    with open('encodings.pickle', 'wb') as f:
        pickle.dump(encoded_faces, f)

# Step 3: Mark Attendance
def mark_attendance(name):
    csv_file = 'attendance.csv'
    if not os.path.exists(csv_file):
        df = pd.DataFrame(columns=['Name', 'Date_Time'])
        df.to_csv(csv_file, index=False)

    df = pd.read_csv(csv_file)
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    df = df.append({'Name': name, 'Date_Time': date_time}, ignore_index=True)
    df.to_csv(csv_file, index=False)

# Step 4: Recognize Faces
def recognize_faces():
    with open('encodings.pickle', 'rb') as f:
        known_face_encodings = pickle.load(f)

    known_face_names = list(known_face_encodings.keys())
    face_encodings = [enc for name in known_face_names for enc in known_face_encodings[name]]

    video_capture = cv2.VideoCapture(0)
    tolerance = 0.4  # Adjusted tolerance for better accuracy

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")  # Use CNN model
        current_face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in current_face_encodings:
            matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance)
            name = "Unknown"

            face_distances = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < tolerance:
                name = known_face_names[best_match_index // len(known_face_encodings[known_face_names[best_match_index]])]
                mark_attendance(name)

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Step 5: Get Attendance
def get_attendance():
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        if not df.empty:
            print("Attendance Records:")
            print(df.to_string(index=False))  # Print the DataFrame as a string without index
        else:
            print("No attendance records found.")
    else:
        print("Attendance file does not exist.")

# Step 6: Export Attendance to Excel
def export_attendance_to_excel():
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        if not df.empty:
            df.to_excel('attendance.xlsx', index=False)
            print("Attendance exported to 'attendance.xlsx'.")
        else:
            print("No attendance records to export.")
    else:
        print("Attendance file does not exist.")

# Step 7: Export Attendance to PDF
def export_attendance_to_pdf():
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        if not df.empty:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Adding a title
            pdf.cell(200, 10, txt="Attendance Report", ln=True, align='C')
            
            # Adding table headers
            pdf.cell(90, 10, 'Name', 1, 0, 'C')
            pdf.cell(90, 10, 'Date_Time', 1, 1, 'C')
            
            # Adding table rows
            for index, row in df.iterrows():
                pdf.cell(90, 10, row['Name'], 1, 0, 'C')
                pdf.cell(90, 10, row['Date_Time'], 1, 1, 'C')
            
            pdf.output("attendance.pdf")
            print("Attendance exported to 'attendance.pdf'.")
        else:
            print("No attendance records to export.")
    else:
        print("Attendance file does not exist.")

# Step 8: Initialize Attendance File (Optional)
if not os.path.exists('attendance.csv'):
    df = pd.DataFrame(columns=['Name', 'Date_Time'])
    df.to_csv('attendance.csv', index=False)

# Main Execution
while True:
    print("\nOptions:\n1. Capture Images\n2. Encode Faces\n3. Recognize Faces and Mark Attendance\n4. Get Attendance\n5. Export Attendance to Excel\n6. Export Attendance to PDF\n7. Exit")
    choice = input("Enter your choice: ")

    if choice == '1':
        employee_name = input("Enter the employee's name: ")
        capture_images(employee_name)
    elif choice == '2':
        encode_faces()
    elif choice == '3':
        recognize_faces()
    elif choice == '4':
        get_attendance()
    elif choice == '5':
        export_attendance_to_excel()
    elif choice == '6':
        export_attendance_to_pdf()
    elif choice == '7':
        break
    else:
        print("Invalid choice. Please try again.")
