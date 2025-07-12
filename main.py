import cv2
from tracker import HandTracker
import numpy, os, json

def loadCameraParams():
    path = f"./img/stereo_calibration.json"
    if not os.path.exists(path):
        print("calibration parameters file not found.")
        exit(1)
    with open(path) as f:
        data = json.load(f)
    rotationMatrix = numpy.array(data["rotation_matrix"], dtype=numpy.float64)
    translationVector = numpy.array(data["translation_vector"], dtype=numpy.float64)
    essentialMatrix = numpy.array(data["essential_matrix"], dtype=numpy.float64)
    fundamentalMatrix = numpy.array(data["fundamental_matrix"], dtype=numpy.float64)
    return rotationMatrix, translationVector, essentialMatrix, fundamentalMatrix
    
def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    #import precalibration data
    
    
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
