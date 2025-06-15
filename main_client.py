import cv2

# IP address of the phone camera.
VIDEO_STREAM_URL = "http://192.168.0.123:5000/video_feed"

cap = cv2.VideoCapture(VIDEO_STREAM_URL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame.")
        break
    cv2.imshow("Phone Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
