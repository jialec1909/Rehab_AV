import cv2
import numpy
from cv2 import aruco

# BOARD = aruco.CharucoBoard((5, 5), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((6, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard((7, 3), 0.030, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def calibrate():
    allCharucoCorners = numpy.array([])
    allCharucoIds = numpy.array([])
    BOARD.setLegacyPattern(True)
    

    # Create a CharucoDetector object
    i = 0
    while i < 20:
        # Grayscale the image
        #frame = cv2.imread(f"img/{i}.png")
        #frame = cv2.imread("1.jpg")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charucoDetector = aruco.CharucoDetector(BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS)

        charucoCorners, charucoIds, markersCorners, markersIds = charucoDetector.detectBoard(gray)

        if (charucoCorners is not None or charucoIds is not None):
            #aruco.interpolateCornersCharuco(markersCorners, markersIds, gray, BOARD, charucoCorners, charucoIds)
            
            frame = aruco.drawDetectedMarkers(frame, corners=markersCorners, ids=markersIds)
            frame = aruco.drawDetectedCornersCharuco(frame, charucoCorners=charucoCorners, charucoIds=charucoIds)
            # Display the image with detected markers
            
        else:
            print("No charuco corners found")
        # Resize the frame to 1/5th of its original size
        newFrame = cv2.resize(frame, (frame.shape[1] // 5, frame.shape[0] // 5))
        cv2.imshow("Charuco Board", newFrame)
        # Wait for the space bar to proceed to the next image
        while True:
            key = cv2.waitKey(0)
            if key == 32:  # Space bar ASCII code
                break
        cv2.destroyAllWindows()

def capture_frame():
    i, j = 0, 0
    # Capture a frame from the camera, 1 frame every 30 frames
    cap = cv2.VideoCapture(1)
    while i < 20:
        ret, frame = cap.read()
        if ret is None:
            print("Failed to capture frame")
            continue
        if j % 30 == 0:
            # save image to img/i.png
            cv2.imwrite(f"img/{i}.png", frame)
            i += 1
        j += 1

    cap.release()



def main():
    #capture_frame()
    calibrate()


if __name__ == "__main__":
    main()
