import cv2
import numpy
from cv2 import aruco
import os
from helper import combineFrame, resizeFrame, matchingCorner
import json

#BOARD = aruco.CharucoBoard((7, 4), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard((8, 6), 0.03, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def createCamera(camID):
    cap = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap

def detect():
    # cap = cv2.VideoCapture(1)
    BOARD.setLegacyPattern(True)

    objPoints = []
    imgPoints0 = []
    imgPoints1 = []

    cap0 = createCamera(0)
    cap1 = createCamera(1)
    i = 0
    j = 0
    cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
    while i < 15:
        path0 = f"./img/stereo/{i}_0.png"
        path1 = f"./img/stereo/{i}_1.png"
        if not os.path.exists(path0) or not os.path.exists(path1):
            _, frame0 = cap0.read()
            _, frame1 = cap1.read() 
        else:
            frame0 = cv2.imread(path0)
            frame1 = cv2.imread(path1)

        if j > 0:
            j -= 1
            continue
        size = frame1.shape
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
        
        if charucoIds0 is not None and charucoIds1 is not None and charucoIds0.size > 4 and charucoIds1.size > 4:
            tempFrame0 = aruco.drawDetectedCornersCharuco(frame0.copy(), charucoCorners0, charucoIds0)
            tempFrame1 = aruco.drawDetectedCornersCharuco(frame1.copy(), charucoCorners1, charucoIds1)

            charucoIds, charucoCorners0, charucoCorners1, = matchingCorner(
                charucoCorners0, charucoIds0, charucoCorners1, charucoIds1)
            temp1, temp2 = BOARD.matchImagePoints(charucoCorners0, charucoIds)
            _, temp3 = BOARD.matchImagePoints(charucoCorners1, charucoIds)
            if len(charucoIds) > 10:
                objPoints.append(temp1)
                imgPoints0.append(temp2)
                imgPoints1.append(temp3)
                i += 1
                print(f"Found {len(charucoIds)} charuco corners on image  {i}_0")
                print(f"Saved image pair {i}")
                #savePicture(frame0, frame1, i)
                #skip the next 30 frames to avoid duplicates
                j = 30
        else:
            tempFrame0 = frame0.copy()
            tempFrame1 = frame1.copy()
        finalFrame = resizeFrame(combineFrame(tempFrame0, tempFrame1))
        cv2.resizeWindow("Camera Feed", int(
            finalFrame.shape[1]), int(finalFrame.shape[0]))
        cv2.imshow("Camera Feed", finalFrame)

    # cap.release()
    cv2.destroyAllWindows()
    return objPoints, imgPoints0, imgPoints1, size[:2]


def savePicture(frame0, frame1, i):
    if not os.path.exists(f"./img/stereo"):
        os.makedirs(f"./img/stereo")
    cv2.imwrite(f"./img/stereo/{i}_0.png", frame0)
    cv2.imwrite(f"./img/stereo/{i}_1.png", frame1)
    print(f"Saved {i}")

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
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
             cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_INTRINSIC)
    R = numpy.zeros((3, 3), dtype=numpy.float64)
    T = numpy.zeros((3, 1), dtype=numpy.float64)
    E = numpy.zeros((3, 3), dtype=numpy.float64)
    F = numpy.zeros((3, 3), dtype=numpy.float64)
    ret = cv2.stereoCalibrate(objPoints, imgPoints0, imgPoint1, cameraMatrix0, distCoeffs0,
                              cameraMatrix1, distCoeffs1, size,  R, T, E, F, flags=flags, criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print("Stereo Calibration Result:")
    print("error:",   str(ret[0]))
    calibration_data = {
        "rotation_matrix": R.tolist(),
        "translation_vector": T.tolist(),
        "essential_matrix": E.tolist(),
        "fundamental_matrix": F.tolist(),
        "camera_intrinsics_0": cameraMatrix0.tolist(),
        "camera_intrinsics_1": cameraMatrix1.tolist(),
        "distortion_coefficients_0": distCoeffs0.tolist(),
        "distortion_coefficients_1": distCoeffs1.tolist()
    }
    with open(f"./img/stereo_calibration.json", "w") as f:
        json.dump(calibration_data, f, indent=4)

def main():
    objPoints, imgPoints0, imgPoints1, size = detect()
    calibrate(objPoints, imgPoints0, imgPoints1, size)

if __name__ == "__main__":
    main()
