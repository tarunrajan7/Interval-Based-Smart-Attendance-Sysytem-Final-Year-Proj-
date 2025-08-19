import cv2

# RTSP with reduced buffering
cap = cv2.VideoCapture(
    "rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101",
    cv2.CAP_FFMPEG
)

# Set buffer size to 1 frame only (important for low latency)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Load YuNet Face Detector
model = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model, "",
    (160, 120),   # much smaller input size → faster
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

screen_res = (1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Downscale frame for detection only
    small_frame = cv2.resize(frame, (320, 240))
    detector.setInputSize((320, 240))
    faces = detector.detect(small_frame)

    # Scale detections back to original frame
    if faces[1] is not None:
        h_ratio = frame.shape[0] / 240
        w_ratio = frame.shape[1] / 320
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:
                # Scale coords to original frame size
                x, y, w, h = int(x * w_ratio), int(y * h_ratio), int(w * w_ratio), int(h * h_ratio)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Resize for display
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))

    cv2.imshow("YuNet Low-Latency", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
