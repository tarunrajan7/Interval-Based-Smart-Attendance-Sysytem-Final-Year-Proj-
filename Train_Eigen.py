import os
import cv2
import numpy as np

# Paths
DATASET_DIR = "Dataset"
MODEL_PATH = "eigen.yml"
FACE_SIZE = (200, 200)  # size for Eigenfaces

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

def prepare_dataset():
    images, labels, names = [], [], []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img = cv2.imread(os.path.join(person_path, fname))
            if img is None:
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            faces = detector.detect(img)

            if faces[1] is not None:
                for face in faces[1]:
                    x, y, fw, fh = map(int, face[:4])
                    face_crop = img[y:y+fh, x:x+fw]
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, FACE_SIZE)
                    images.append(gray)
                    labels.append(label_id)

        label_id += 1

    return images, np.array(labels), label_map

print("Preparing dataset...")
images, labels, label_map = prepare_dataset()

print("Training Eigenfaces recognizer...")
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.train(images, labels)
recognizer.save(MODEL_PATH)

# Save label map
np.save("label_map.npy", label_map)
print("Training complete. Model saved as", MODEL_PATH)
