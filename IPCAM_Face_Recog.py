import cv2
import face_recognition

# Load your known face
known_image = face_recognition.load_image_file("tarun.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_encodings = [known_encoding]
known_names = ["Tarun"]

# Lab camera RTSP stream
video_capture = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Desired display resolution
screen_res = 1280, 720

process_this_frame = True

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from lab camera")
        break

    # Resize frame to 50% for faster CNN detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        # CNN face detection for accuracy
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame  # skip every 2nd frame

    # Draw boxes and resize to fit screen while keeping aspect ratio
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up to match original frame
        scale = 2  # since fx=0.5
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # --- Resize frame to fit screen while keeping aspect ratio ---
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    display_frame = cv2.resize(frame, (window_width, window_height))
    # -------------------------------------------------------------

    cv2.imshow('Lab Camera Face Recognition (CNN)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
