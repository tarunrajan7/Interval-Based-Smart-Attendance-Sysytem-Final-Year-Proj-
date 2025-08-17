import cv2
import face_recognition
import numpy as np

# ========================
# 1. Load Known Faces
# ========================
# You can replace these with your own images
known_face_encodings = []
known_face_names = []

# Example: load two sample images
tarun_img = face_recognition.load_image_file("tarun.jpg")
tarun_encoding = face_recognition.face_encodings(tarun_img)[0]
known_face_encodings.append(tarun_encoding)
known_face_names.append("Tarun")

#friend_img = face_recognition.load_image_file("friend.jpg")
#friend_encoding = face_recognition.face_encodings(friend_img)[0]
#known_face_encodings.append(friend_encoding)
#known_face_names.append("Friend")

# ========================
# 2. Camera Input
# ========================
# For Webcam (default):
cap = cv2.VideoCapture(0)

# For Hikvision RTSP stream (uncomment below):
# cap = cv2.VideoCapture("rtsp://username:password@IP:554/Streaming/Channels/101")

# ========================
# 3. Processing Loop
# ========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect all faces + encodings
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")  # "hog" for faster, "cnn" for accurate
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop over detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # Use the best match (smallest distance)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Label
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
