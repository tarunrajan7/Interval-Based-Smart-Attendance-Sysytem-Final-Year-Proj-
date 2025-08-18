import cv2

# RTSP URL (update if needed)
cap = cv2.VideoCapture("rtsp://test:Test@123@192.168.101.63:554/Streaming/Channels/2101")

# Set desired screen resolution
screen_res = 1920, 1080 

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
    # --------------------------------------------------------

    cv2.imshow("Lab Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
