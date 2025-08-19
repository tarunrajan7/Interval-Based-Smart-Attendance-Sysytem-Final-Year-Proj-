import cv2
import face_recognition

# --- Known face(s) ---
known_image = face_recognition.load_image_file("tarun.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
known_encodings = [known_encoding]
known_names = ["Tarun"]

# --- RTSP (with low-latency hints) ---
rtsp = "rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101"
cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)   # use FFMPEG backend
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # drop old buffered frames

# --- Target window size (no 'zoom') ---
screen_res = (1280, 720)  # (width, height)

process_this_frame = True
face_locations, face_names = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === FAST PATH for detection/recognition (work on a smaller copy) ===
    small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for fe in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, fe, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            face_names.append(name)

    process_this_frame = not process_this_frame  # skip every 2nd frame for speed

    # === Draw on the ORIGINAL frame (correct scale back) ===
    scale_back = 1 / 0.75  # = 4/3
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top    = int(top    * scale_back)
        right  = int(right  * scale_back)
        bottom = int(bottom * scale_back)
        left   = int(left   * scale_back)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # === Letterbox to fit 1280x720 WITHOUT cropping/zooming ===
    h, w = frame.shape[:2]
    sw, sh = screen_res
    scale = min(sw / w, sh / h)
    disp_w, disp_h = int(w * scale), int(h * scale)
    display_frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("Face Recognition (No Zoom)", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
