import cv2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# RTSP stream
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Desired screen resolution
screen_res = (1280, 720)
frame_count = 0
faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Resize to fit screen while keeping aspect ratio ---
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    frame = cv2.resize(frame, (window_width, window_height))
    # -------------------------------------------------------

    frame_count += 1

    # Run detection every 3rd frame (reduces latency)
    if frame_count % 3 == 0:
        # Smaller frame for faster detection
        small_frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,   # faster but less accurate
            minNeighbors=5,
            minSize=(30, 30)   # detect smaller faces
        )

        # Scale face coordinates back to original frame size
        scale_back_x = frame.shape[1] / small_frame.shape[1]
        scale_back_y = frame.shape[0] / small_frame.shape[0]
        faces = [(int(x * scale_back_x), int(y * scale_back_y),
                  int(w * scale_back_x), int(h * scale_back_y)) for (x, y, w, h) in detected_faces]

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Lab Camera - Optimized Haar Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
