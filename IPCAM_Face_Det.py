import cv2
import threading

# RTSP Camera URL
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Load YuNet model
model = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model, "",
    (320, 240),  # smaller input size → faster
    score_threshold=0.8,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Desired display resolution
screen_res = (1280, 720)

# Threaded frame grabber
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

vs = VideoStream("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

frame_count = 0
faces = None

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Run detection every 5 frames only
    if frame_count % 5 == 0:
        faces = detector.detect(frame)

    # Draw faces
    if faces is not None and faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = map(int, face[:4])
            conf = face[-1]
            if conf > 0.8:
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

    cv2.imshow("YuNet Face Detection (Optimized)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
