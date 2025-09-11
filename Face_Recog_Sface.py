import cv2
import threading
import os
import numpy as np

# ------------------- Paths -------------------
dataset_path = "Dataset"
det_model = "face_detection_yunet_2023mar.onnx"
rec_model = "face_recognition_sface_2021dec.onnx"

# ------------------- Load Models -------------------
detector = cv2.FaceDetectorYN.create(
    det_model, "",
    (320, 240),
    score_threshold=0.85,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

recognizer = cv2.FaceRecognizerSF.create(
    rec_model, "",
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# ------------------- Load Known Faces -------------------
known_features = []
known_names = []

print("[INFO] Loading dataset...")
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        detector.setInputSize((w, h))
        faces = detector.detect(image)

        if faces[1] is not None:
            for face in faces[1]:
                # Align + Extract features
                aligned = recognizer.alignCrop(image, face)
                feature = recognizer.feature(aligned)

                known_features.append(feature)
                known_names.append(person_name)
        else:
            print(f"[WARN] No face found in {img_path}")

print(f"[INFO] Loaded {len(known_features)} embeddings for {set(known_names)}")

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

    if frame_count % 3 == 0:  # detect every 3rd frame
        faces = detector.detect(frame)

    if faces is not None and faces[1] is not None:
        for face in faces[1]:
            x, y, fw, fh = map(int, face[:4])
            conf = face[-1]

            if conf > 0.85:
                aligned_face = recognizer.alignCrop(frame, face)
                feature = recognizer.feature(aligned_face)

                name = "Unknown"
                best_score = -1

                for db_feature, db_name in zip(known_features, known_names):
                    # Compare features using cosine similarity
                    score = recognizer.match(feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > best_score:
                        best_score = score
                        name = db_name

                # Threshold tuning (0.5-0.6 works well for cosine)
                if best_score < 0.5:
                    name = "Unknown"

                # Draw bounding box + label
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({best_score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Resize for display
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet + SFace Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
