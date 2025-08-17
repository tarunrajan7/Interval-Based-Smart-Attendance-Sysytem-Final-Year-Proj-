import cv2
import face_recognition

# Load your known face
known_image = face_recognition.load_image_file("tarun.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_encodings = [known_encoding]
known_names = ["Tarun"]

# Webcam
video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize less (0.75 instead of 0.25)
    small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame  # skip every 2nd frame for speed

    # Draw boxes
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up
        scale = 4/3  # since fx=0.75
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
