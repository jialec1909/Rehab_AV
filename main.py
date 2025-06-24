
import cv2
from tracker import HandTracker

def main():
    
    cap0 = cv2.VideoCapture(0)
    #standardize cam res to 720p
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1 = cv2.VideoCapture(1)
    #standardize cam res to 720p
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    tracker0 = HandTracker()
    tracker1 = HandTracker()

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        frame0 = tracker0.process_frame(frame0)
        tracker0.draw_trajectory_smooth(frame0)
        frame1 = tracker1.process_frame(frame1)
        tracker1.draw_trajectory_smooth(frame1)

        cv2.imshow("Rehab Tracker 0", frame0)
        cv2.imshow("Rehab Tracker 1", frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

