import cv2
import threading
import face_recognition
import numpy as np
import os

# ----------------- RTSP Camera -----------------
# Use 0 for webcam, or your RTSP URL
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

# ----------------- Known Faces (Multi-Posture) -----------------
known_encodings = []
known_names = []

def _list_images(folder_dir, exts=(".jpg", ".jpeg", ".png")):
    paths = []
    for root, _, files in os.walk(folder_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths

def _yunet_detect(img_bgr):
    """Return faces ndarray using YuNet; handles OpenCV return shapes."""
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    result = detector.detect(img_bgr)
    if isinstance(result, (tuple, list)):
        faces = result[1]
    else:
        faces = result
    return faces

def add_person(name, folder_dir):
    """
    Add all images of a person from a folder.
    Example: add_person("Tarun", r"T:\TARUN\EIE\Final Yr Proj\Tarun_Faces")
    """
    images = _list_images(folder_dir)
    print(f"[INFO] Loading {len(images)} images for {name}")

    added = 0
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load {img_path}")
            continue

        # 1) Try YuNet detect -> crop
        faces = _yunet_detect(img)

        face_crop = None
        if faces is not None and len(faces) > 0:
            x, y, w, h = faces[0][:4].astype(int)
            x = max(0, x); y = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            if x2 > x and y2 > y:
                face_crop = img[y:y2, x:x2]

        # 2) Fallback: use face_recognition CNN detector on the full image
        if face_crop is None or face_crop.size == 0:
            rgb_full = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb_full, model="cnn")
            if len(boxes) > 0:
                top, right, bottom, left = boxes[0]
                face_crop = img[top:bottom, left:right]

        if face_crop is None or face_crop.size == 0:
            print(f"[WARNING] No face detected in {os.path.basename(img_path)}")
            continue

        # Encode from the crop by telling face_recognition that the entire crop is a face
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        fh, fw = face_rgb.shape[:2]
        encs = face_recognition.face_encodings(
            face_rgb,
            known_face_locations=[(0, fw, fh, 0)],  # (top, right, bottom, left) in the cropped image
            num_jitters=1,
            model="small"
        )

        if len(encs) > 0:
            known_encodings.append(encs[0])
            known_names.append(name)
            added += 1
            print(f"[OK] Added {os.path.basename(img_path)}")
        else:
            print(f"[WARNING] No encoding from {os.path.basename(img_path)}")

    print(f"[INFO] Added {added}/{len(images)} encodings for {name}")

# ✅ Add your folder here (supports .jpg + .jpeg + .png)
add_person("Tarun", r"T:\TARUN\EIE\Final Yr Proj\Tarun_Faces")

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
    x = max(0, x); y = max(0, y)
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    face_img = frame[y:y2, x:x2]
    if face_img.size == 0:
        return "Unknown", 0.0

    # Encode from the crop directly (bypass re-detection inside dlib)
    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    fh, fw = rgb_face.shape[:2]
    encodings = face_recognition.face_encodings(
        rgb_face,
        known_face_locations=[(0, fw, fh, 0)],
        num_jitters=1,
        model="small"
    )

    if len(encodings) == 0 or len(known_encodings) == 0:
        return "Unknown", 0.0

    encoding = encodings[0]
    # 0.6 is typical; you used 0.5 earlier (stricter). Keep 0.5 if you want fewer false accepts.
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.55)
    face_distances = face_recognition.face_distance(known_encodings, encoding)

    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            # Convert distance to a crude confidence
            conf = max(0.0, 1.0 - float(face_distances[best_match_index]))
            return known_names[best_match_index], conf

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
        result = detector.detect(frame)
        faces = result[1] if isinstance(result, (tuple, list)) else result

    if faces is not None:
        for face in faces:
            x, y, ww, hh = map(int, face[:4])
            conf = float(face[-1]) if len(face) >= 15 else 1.0  # YuNet puts score at the end
            if conf > 0.8:
                name, score = recognize_face(frame, face)
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("YuNet + FaceRecognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
