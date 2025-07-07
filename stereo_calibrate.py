import cv2
import numpy
from cv2 import aruco
import os
from time import sleep
import json

# BOARD = aruco.CharucoBoard((5, 5), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((6, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((7, 3), 0.030, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard(
    (11, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
)
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def detect():
    # cap = cv2.VideoCapture(1)
    BOARD.setLegacyPattern(True)

    objPoints = []
    imgPoints0 = []
    imgPoints1 = []

    # get count of pictures in /img
    imgCount = len(os.listdir(f"./img/stereo/"))/2  # halves
    for i in range(int(imgCount)):
        # frame = BOARD.generateImage((900,900),10,1)
        frame0 = cv2.imread(f"./img/stereo/{i}_0.png")
        frame1 = cv2.imread(f"./img/stereo/{i}_1.png")
        size = frame0.shape
        # Grayscale the image
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        charucoDetector = aruco.CharucoDetector(
            BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS)

        charucoCorners0, charucoIds0, _, _ = (
            charucoDetector.detectBoard(gray0)
        )
        charucoCorners1, charucoIds1, _, _ = (
            charucoDetector.detectBoard(gray1)
        )

        if charucoIds0 is not None and charucoIds1 is not None and charucoIds0.size > 4 and charucoIds1.size > 4 and charucoIds0.size == charucoIds1.size:
            temp1, temp2 = BOARD.matchImagePoints(charucoCorners0, charucoIds0)
            _, temp3 = BOARD.matchImagePoints(charucoCorners1, charucoIds1)
            print(f"Found {len(temp1)} charuco corners on image  {i}_0")
            print(f"Found {len(temp3)} charuco corners on image  {i}_1")
            objPoints.append(temp1)
            imgPoints0.append(temp2)
            imgPoints1.append(temp3)
        else:
            print("No charuco corners found on image pair " + str(i))

    # cap.release()
    cv2.destroyAllWindows()
    return objPoints, imgPoints0, imgPoints1, size[:2]


def createCamera(camID):
    cap = cv2.VideoCapture(camID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def takePicture():
    if not os.path.exists(f"./img/stereo"):
        os.makedirs(f"./img/stereo")
    else:
        print(f"Directory ./img/stereo already exists. Pictures will be overwritten.")
    i = 0
    cap0 = createCamera(0)
    cap1 = createCamera(1)
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        combined = numpy.vstack((frame0, frame1))
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Feed", 960, 1080)
        cv2.imshow("Camera Feed", combined)
        if not ret0 or not ret1:
            print("Failed to grab frame")
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(f"./img/stereo/{i}_0.png", frame0)
            cv2.imwrite(f"./img/stereo/{i}_1.png", frame1)
            print(f"Saved {i}")
            i += 1
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


def loadCameraParams(camID):
    path = f"./img/camera_calibration_{camID}.json"
    if not os.path.exists(path):
        print("Camera parameters file not found.")
        exit(1)
    with open(path) as f:
        data = json.load(f)
    cameraMatrix = numpy.array(data["camera_matrix"], dtype=numpy.float64)
    distCoeffs = numpy.array(
        data["distortion_coefficients"], dtype=numpy.float64)
    return cameraMatrix, distCoeffs


def calibrate(objPoints, imgPoints0, imgPoint1, size):

    cameraMatrix0, distCoeffs0 = loadCameraParams(0)
    cameraMatrix1, distCoeffs1 = loadCameraParams(1)
    flags = cv2.CALIB_FIX_INTRINSIC
    R = numpy.zeros((3, 3), dtype=numpy.float64)
    T = numpy.zeros((3, 1), dtype=numpy.float64)
    E = numpy.zeros((3, 3), dtype=numpy.float64)
    F = numpy.zeros((3, 3), dtype=numpy.float64)
    ret = cv2.stereoCalibrate(objPoints, imgPoints0, imgPoint1, cameraMatrix0, distCoeffs0,
                                                      cameraMatrix1, distCoeffs1, size,  R, T, E, F, flags=flags)
    print("Stereo Calibration Result:")
    print("error:",   str(ret[0]))
    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", T)
    print("Essential Matrix:\n", E)
    print("Fundamental Matrix:\n", F)
    #print("Rvec:\n", rvec)
    #print("Tvec:\n", tvec)


def StereoCameraCalibration():
    if not os.path.exists(f"./img/stereo/0_1.png"):
        takePicture()
    objPoints, imgPoints0, imgPoints1, size = detect()
    return calibrate(objPoints, imgPoints0, imgPoints1, size)


def main():

    StereoCameraCalibration()


if __name__ == "__main__":
    main()
