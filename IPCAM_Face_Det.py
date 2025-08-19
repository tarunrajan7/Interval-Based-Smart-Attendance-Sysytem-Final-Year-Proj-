import cv2

# Load YuNet model
model_path = "face_detection_yunet_2023mar.onnx"   # keep the model in same folder
face_detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(320, 320),   # detection input size (lower = faster, higher = more accurate)
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

# RTSP Camera URL
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Set desired display resolution
screen_res = (1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces (use original frame size for detection)
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    faces = face_detector.detect(frame)[1]

    # Draw detections
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- Resize to fit screen while keeping aspect ratio ---
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame_resized = cv2.resize(frame, (window_width, window_height))
    # --------------------------------------------------------

    cv2.imshow("Lab Camera Stream", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
