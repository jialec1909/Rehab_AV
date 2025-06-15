import cv2
from tracker import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process_frame(frame)
        #tracker.draw_trajectory_pseudo3D(frame)
        tracker.draw_trajectory_smooth(frame)

        cv2.imshow("Rehab Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
