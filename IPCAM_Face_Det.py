import cv2

# RTSP Camera URL
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Load YuNet Face Detector
model = "face_detection_yunet_2023mar.onnx"  # download from OpenCV Zoo if not present
detector = cv2.FaceDetectorYN.create(
    model, "",
    (320, 240),   # input size (smaller = faster, larger = more accurate)
    score_threshold=0.8,  # confidence filter
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Desired display resolution
screen_res = (1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster detection (not display)
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Detect faces
    faces = detector.detect(frame)
    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:  # filter low-confidence boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize for display (to avoid zoom issue)
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
