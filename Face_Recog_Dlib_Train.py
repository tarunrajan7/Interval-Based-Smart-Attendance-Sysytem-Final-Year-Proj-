import cv2
import dlib
import numpy as np
import os

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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect_dlib = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
    shape = predictor(gray, rect_dlib)
    face_chip = dlib.get_face_chip(img, shape, size=150)
    return np.array(face_rec_model.compute_face_descriptor(face_chip))

# ------------------- Build Dataset -------------------
def build_dlib_dataset(dataset_path):
    embeddings = []
    labels = []

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

    return np.array(embeddings), np.array(labels)

dataset_path = "Dataset"
embeddings, labels = build_dlib_dataset(dataset_path)

# Save to file
np.savez("dlib_embeddings.npz", embeddings=embeddings, labels=labels)
print("✅ Training complete. Saved to dlib_embeddings.npz")
