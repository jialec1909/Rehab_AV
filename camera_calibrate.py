import cv2
import numpy
from cv2 import aruco
import os
from time import sleep

# BOARD = aruco.CharucoBoard((5, 5), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((6, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((7, 3), 0.030, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard(
    (11, 8), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
)
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def detect():
    # cap = cv2.VideoCapture(1)
    allCharucoCorners = numpy.array([])
    allCharucoIds = numpy.array([])
    BOARD.setLegacyPattern(True)

    # get count of pictures in /img
    imgCount = len(os.listdir("img"))
    for i in range(imgCount):
        # frame = BOARD.generateImage((900,900),10,1)
        frame = cv2.imread("./img/charuco_capture_" + str(i) + ".png")
        size = frame.shape
        # ret, frame = cap.read()
        # if not ret:
        #    print("Failed to grab frame")
        #    break

        # Grayscale the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charucoDetector = aruco.CharucoDetector(
            BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS)

        charucoCorners, charucoIds, markersCorners, markersIds = (
            charucoDetector.detectBoard(gray)
        )

        if charucoCorners is not None or charucoIds is not None:
            # aruco.interpolateCornersCharuco(markersCorners, markersIds, gray, BOARD, charucoCorners, charucoIds)

            newImg = aruco.drawDetectedMarkers(
                image=frame.copy(), corners=markersCorners, ids=markersIds
            )
            newImg = aruco.drawDetectedCornersCharuco(
                image=newImg, charucoCorners=charucoCorners, charucoIds=charucoIds
            )

            # Display the image with detected markers
            cv2.imshow("Charuco Board", newImg)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            
            try:
                calibrate(charucoCorners, charucoIds, size, i)
            except Exception as e:
                print(f"Error during calibration: {e}")
        else:
            print("No charuco corners found")

    # cap.release()
    cv2.destroyAllWindows()


def takePicture():
    i = 0
    cap = cv2.VideoCapture(1)
    while i < 10:
        ret, frame = cap.read()
        cv2.imshow("Camera Feed", frame)
        if not ret:
            print("Failed to grab frame")
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"./img/charuco_capture_{i}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            i += 1
    cap.release()
    cv2.destroyAllWindows()


def calibrate(charucoCorners, charucoIds, size, i):
            objPoints, imgPoints = BOARD.matchImagePoints(
                charucoCorners, charucoIds
            )

            if objPoints is None and imgPoints is None:
                return

            cameraMatrix = numpy.zeros((3, 3), dtype=numpy.float64)
            distCoeffs = numpy.zeros((5, 1), dtype=numpy.float64)
            rvecs = []
            tvecs = []
            error = 0.0
            cv2.calibrateCameraExtended(
                objPoints, imgPoints, (size[0], size[1]), cameraMatrix, distCoeffs, rvecs, tvecs, perViewErrors=error
            )

            print(f"Calibration error for image {i}: {error}")

def main():
    #takePicture()

    detect()
    print("testing calibration")


if __name__ == "__main__":
    main()
