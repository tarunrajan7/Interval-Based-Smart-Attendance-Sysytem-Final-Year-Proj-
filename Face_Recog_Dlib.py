import cv2
import dlib
import numpy as np
import os
import threading

# ------------------- Dlib Models -------------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)

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

# ------------------- Embedding Extractor -------------------
def get_face_embedding(img, rect):
    """Align face with landmarks & get 128D embedding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect_dlib = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
    shape = predictor(gray, rect_dlib)
    face_chip = dlib.get_face_chip(img, shape, size=150)
    return np.array(face_rec_model.compute_face_descriptor(face_chip))

# ------------------- Dataset Loader -------------------
def build_dlib_dataset(dataset_path):
    embeddings = []
    labels = []
    names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            faces = detector.detect(img)

            if faces is not None and faces[1] is not None:
                for face in faces[1]:
                    x, y, fw, fh = map(int, face[:4])
                    emb = get_face_embedding(img, (x, y, x+fw, y+fh))
                    embeddings.append(emb)
                    labels.append(person_name)
                    names.append(person_name)
    return np.array(embeddings), np.array(labels), list(set(names))

dataset_path = "Dataset"  # your dataset folder
embeddings, labels, unique_names = build_dlib_dataset(dataset_path)

# ------------------- Recognizer -------------------
def recognize_face(face_emb, threshold=0.6):
    if len(embeddings) == 0:
        return "Unknown"

    distances = np.linalg.norm(embeddings - face_emb, axis=1)
    min_dist = np.min(distances)
    min_idx = np.argmin(distances)

    if min_dist < threshold:
        return labels[min_idx]
    else:
        return "Unknown"

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

# Use RTSP or webcam
vs = VideoStream(0)

screen_res = (1280, 720)
frame_count = 0
faces_detected = None

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
                face_emb = get_face_embedding(frame, (x, y, x+fw, y+fh))
                name = recognize_face(face_emb)

                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                cv2.putText(frame, f"Dlib: {name}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet + Dlib Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
