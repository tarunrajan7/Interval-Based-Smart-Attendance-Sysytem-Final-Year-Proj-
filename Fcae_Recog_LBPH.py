import cv2
import dlib
import numpy as np
import os

# Load Dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ------------------- Step 1: Encode dataset -------------------
def encode_face_dataset(dataset_path="Dataset"):
    embeddings, names = [], []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            dets = detector(rgb_img, 1)
            for det in dets:
                shape = sp(rgb_img, det)
                face_descriptor = facerec.compute_face_descriptor(rgb_img, shape)
                embeddings.append(np.array(face_descriptor))
                names.append(person_name)

    return embeddings, names

known_embeddings, known_names = encode_face_dataset("Dataset")

# ------------------- Step 2: Recognition -------------------
def recognize_face(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_frame, 1)

    for det in dets:
        shape = sp(rgb_frame, det)
        face_descriptor = facerec.compute_face_descriptor(rgb_frame, shape)
        face_embedding = np.array(face_descriptor)

        # Compare with known embeddings
        distances = np.linalg.norm(known_embeddings - face_embedding, axis=1)
        min_index = np.argmin(distances)
        if distances[min_index] < 0.6:  # threshold
            name = known_names[min_index]
        else:
            name = "Unknown"

        # Draw results
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

# ------------------- Step 3: Run Live Camera -------------------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_face(frame)
    cv2.imshow("Dlib Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
