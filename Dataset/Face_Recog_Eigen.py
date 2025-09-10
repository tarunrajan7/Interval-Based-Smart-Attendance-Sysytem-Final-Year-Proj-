import cv2
import threading
import numpy as np

# Load trained model
MODEL_PATH = "eigen.yml"
LABEL_MAP_PATH = "label_map.npy"
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read(MODEL_PATH)
label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()

# RTSP Camera URL
RTSP_URL = 0

# Load YuNet
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

FACE_SIZE = (200, 200)

# Threaded frame grabber
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

    # Run detection every 5 frames
    if frame_count % 5 == 0:
        faces = detector.detect(frame)

    if faces is not None and faces[1] is not None:
        for face in faces[1]:
            x, y, fw, fh = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:
                face_crop = frame[y:y+fh, x:x+fw]
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, FACE_SIZE)

                label_id, confidence = recognizer.predict(gray)
                name = label_map.get(label_id, "Unknown")

                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.1f})", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("YuNet + Eigenfaces Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
