import cv2

# Correct RTSP URL
rtsp_url = "rtsp://test:Test%40123@192.168.101.63:554/Streaming/Channels/2101"

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Lab Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
