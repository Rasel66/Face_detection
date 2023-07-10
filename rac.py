import cv2
import face_recognition
import os
import datetime
import pandas as pd


def take_images():
    names = []
    images = []

    num_images = int(input("Enter the number of images you want to capture: "))

    for i in range(num_images):
        name = input("Enter the name of person {}: ".format(i+1))
        names.append(name)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Unable to open the webcam")
            exit()

        while True:
            ret, frame = cap.read()
            cv2.imshow('Capture Images', frame)

            if cv2.waitKey(1) == ord('p'):
                images.append(frame)
                break
            elif cv2.waitKey(1) == ord('q'):
                break


        cap.release()
        cv2.destroyAllWindows()

    folder_path = "person/images"  
    os.makedirs(folder_path, exist_ok=True) 

    for i, image in enumerate(images):
        file_name = "{}.jpg".format(names[i]) 
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, image)

    print("Images captured successfully!")
    detect_person()

    return images, names


def load_known_faces(folder_path):
    known_images = []
    known_names = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(folder_path, file_name)
            known_images.append(image_path)
            name = os.path.splitext(file_name)[0]
            known_names.append(name)

    known_faces = []
    for image_path in known_images:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            known_faces.append(face_encoding)
        else:
            print("No face detected in image: {}".format(image_path))

    return known_names, known_faces

def detect_person():
    folder_path = "person/images" 

    known_names, known_faces = load_known_faces(folder_path)

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Unable to open the webcam")
        exit()

    data = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if the Excel file exists
    if os.path.isfile("person_entry_log.xlsx"):
        # Load previous data from the Excel file
        data = pd.read_excel("person_entry_log.xlsx")

    while True:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        distance_threshold = 0.4

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            min_distance = min(face_distances)

            if min_distance <= distance_threshold:
                # Face is recognized
                min_distance_index = list(face_distances).index(min_distance)
                name = known_names[min_distance_index]

                current_date = datetime.date.today().strftime("%Y-%m-%d")
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                new_entry = pd.DataFrame({"Name": [name], "Date": [current_date], "Time": [current_time]})
                data = pd.concat([data, new_entry], ignore_index=True)
                data.to_excel("person_entry_log.xlsx", index=False)
            else:
                # Face is unknown
                name = "Unknown"

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (0, 0, 0), 1)

        cv2.imshow('Person Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# # Main menu
def main_menu():
    print("Options:")
    print("1. Take images and their names")
    print("2. Detect the person")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        take_images()
    elif choice == "2":
        detect_person()
    else:
        print("Invalid choice. Please try again.")
        main_menu()

# Start the program
main_menu()
