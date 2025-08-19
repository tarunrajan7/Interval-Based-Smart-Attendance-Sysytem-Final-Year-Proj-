import cv2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# RTSP stream from Hikvision camera
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Desired screen resolution
screen_res = (1280, 720)

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

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Lab Camera - Haar Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
