import cv2

# Replace with your RTSP link
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to fit your screen (example: 1280x720)
    frame = cv2.resize(frame, (1280, 720))

    cv2.imshow("Lab Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
