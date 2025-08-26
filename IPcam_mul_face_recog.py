import cv2
import threading
import numpy as np

# ----------------- LBPH Recognizer -----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r"T:\TARUN\EIE\Final Yr Proj\lbph_model.yml")

# Label mapping from your training script
labels = {0: "Rithesh", 1: "Tarun"}

# ----------------- RTSP / Webcam -----------------
RTSP_URL = 0  # use 0 for webcam, or replace with RTSP URL

# ----------------- YuNet Detector -----------------
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
        result = detector.detect(frame)
        faces = result[1] if isinstance(result, (tuple, list)) else result

    if faces is not None:
        for face in faces:
            x, y, ww, hh = map(int, face[:4])
            conf = float(face[-1]) if len(face) >= 15 else 1.0
            if conf > 0.8:
                # Crop + preprocess for LBPH
                face_roi = frame[y:y+hh, x:x+ww]
                if face_roi.size > 0:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (200, 200))
                    gray_face = cv2.equalizeHist(gray_face)

                    label_id, confidence = recognizer.predict(gray_face)

                    # Lower confidence = better match
                    if confidence < 70:  # threshold, tune if needed
                        name = labels.get(label_id, "Unknown")
                    else:
                        name = "Unknown"

                    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.1f})",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

    # Resize display window
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("YuNet + LBPH Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
