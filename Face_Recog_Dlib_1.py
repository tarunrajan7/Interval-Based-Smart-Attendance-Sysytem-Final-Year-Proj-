import cv2
import threading
import os
import face_recognition

# ------------------- Load Known Faces -------------------
dataset_path = "Dataset"  # Folder with subfolders for each person
known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

print(f"[INFO] Loaded {len(known_encodings)} face encodings for {set(known_names)}")

# ------------------- YuNet Face Detector -------------------
model = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model, "",
    (320, 240),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# ------------------- RTSP Video Stream -------------------
class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# Replace with your RTSP or webcam
vs = VideoStream("rtsp://test:Test@123@192.168.101.72:554/Streaming/Channels/2101")

# ------------------- Main Loop -------------------
screen_res = (1280, 720)
frame_count = 0
faces = None

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    if frame_count % 5 == 0:  # detect every 5th frame
        faces = detector.detect(frame)

    if faces is not None and faces[1] is not None:
        for face in faces[1]:
            x, y, fw, fh = map(int, face[:4])
            conf = face[-1]

            if conf > 0.8:
                # Crop face ROI
                face_roi = frame[y:y+fh, x:x+fw]
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb_face)
                name = "Unknown"

                if len(encodings) > 0:
                    matches = face_recognition.compare_faces(known_encodings, encodings[0])
                    face_distances = face_recognition.face_distance(known_encodings, encodings[0])
                    best_match_index = face_distances.argmin()

                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                # Draw bounding box + label
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({conf:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize for display
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet + Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
