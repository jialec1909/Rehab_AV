import cv2
import numpy
from cv2 import aruco

# BOARD = aruco.CharucoBoard((5, 5), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((6, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard((7, 3), 0.030, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def calibrate():
    cap = cv2.VideoCapture(1)
    allCharucoCorners = numpy.array([])
    allCharucoIds = numpy.array([])
    BOARD.setLegacyPattern(True)
    
    # Create a CharucoDetector object
    
    while True:
        #frame = BOARD.generateImage((900,900),10,1)
        #frame = cv2.imread("1.jpg")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Grayscale the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charucoDetector = aruco.CharucoDetector(BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS)

        charucoCorners, charucoIds, markersCorners, markersIds = charucoDetector.detectBoard(gray)

        if (charucoCorners is not None or charucoIds is not None):
            #aruco.interpolateCornersCharuco(markersCorners, markersIds, gray, BOARD, charucoCorners, charucoIds)
            
            newImg = aruco.drawDetectedMarkers(image=gray.copy(), corners=markersCorners, ids=markersIds)
            newImg = aruco.drawDetectedCornersCharuco(image=newImg, charucoCorners=charucoCorners, charucoIds=charucoIds)

            # Display the image with detected markers
            cv2.imshow("Charuco Board", newImg)
            cv2.waitKey(1)
        else:
            print("No charuco corners found")
    #cap.release()
    cv2.destroyAllWindows()


def main():
    calibrate()


if __name__ == "__main__":
    main()
