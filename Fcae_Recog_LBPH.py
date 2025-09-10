import cv2
import os
import numpy as np
import threading

# ------------------- Dataset Loader -------------------
def load_dataset(dataset_path):
    faces, labels = [], []
    label_map, current_label = {}, 0

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name not in label_map:
            label_map[person_name] = current_label
            current_label += 1

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_map[person_name])

    return faces, np.array(labels), {v: k for k, v in label_map.items()}

dataset_path = "Dataset"  # adjust to your dataset folder
faces, labels, label_map = load_dataset(dataset_path)

# ------------------- Train LBPH Recognizer -------------------
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, labels)

# ------------------- YuNet Detector -------------------
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

vs = VideoStream(0)

screen_res = (1280, 720)
frame_count = 0
faces_detected = None

# ------------------- Recognition -------------------
def recognize_lbph(face_gray):
    label_id, confidence = lbph.predict(face_gray)
    if confidence > 80:  # Threshold for Unknown
        return "Unknown"
    return label_map.get(label_id, "Unknown")

# ------------------- Main Loop -------------------
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    if frame_count % 5 == 0:
        faces_detected = detector.detect(frame)

    if faces_detected is not None and faces_detected[1] is not None:
        for face in faces_detected[1]:
            x, y, fw, fh = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:
                face_gray = cv2.cvtColor(frame[y:y+fh, x:x+fw], cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (faces[0].shape[1], faces[0].shape[0]))

                name = recognize_lbph(face_resized)

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.putText(frame, f"LBPH: {name}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet + LBPH Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
