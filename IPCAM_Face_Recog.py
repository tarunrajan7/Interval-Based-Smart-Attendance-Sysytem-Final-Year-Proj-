import cv2
import threading
import face_recognition
import numpy as np

# ----------------- RTSP Camera -----------------
RTSP_URL = "rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101"

# ----------------- Load YuNet Detector -----------------
det_model = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    det_model, "",
    (320, 240),
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# ----------------- Known Faces (using face_recognition) -----------------
known_encodings = []
known_names = []

# Example: Add your known face
img = face_recognition.load_image_file("tarun.jpg")
encoding = face_recognition.face_encodings(img)[0]
known_encodings.append(encoding)
known_names.append("Tarun")

# ----------------- Threaded Frame Grabber -----------------
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

# ----------------- Helper: Recognize using embeddings -----------------
def recognize_face(frame, face_box):
    x, y, w, h = map(int, face_box[:4])
    face_img = frame[y:y+h, x:x+w]

    # Convert to RGB for face_recognition
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb_face)
    if len(encodings) == 0:
        return "Unknown", 0.0

    encoding = encodings[0]

    # Compare with database
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, encoding)

    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return known_names[best_match_index], 1 - face_distances[best_match_index]

    return "Unknown", 0.0

# ----------------- Main Loop -----------------
vs = VideoStream(RTSP_URL)
frame_count = 0
faces = None

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Detect every 5 frames for speed
    if frame_count % 5 == 0:
        faces = detector.detect(frame)

    if faces is not None and faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:
                name, score = recognize_face(frame, face)

                # Draw bounding box + label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    # Resize for display
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("YuNet + FaceRecognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
