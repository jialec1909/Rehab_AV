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
    (8, 6), 0.03, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
)
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()


def createCamera(camID):
    cap = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def combineFrame(frame0, frame1):
    if frame0.shape[0] >= frame1.shape[0]:
        newWidth = int(frame0.shape[1] / frame0.shape[0] * frame1.shape[0])
        frame0 = cv2.resize(frame0, (newWidth, frame1.shape[0]))
    else:
        newWidth = int(frame1.shape[1] / frame1.shape[0] * frame0.shape[0])
        frame1 = cv2.resize(frame1, (newWidth, frame0.shape[0]))
    cv2.waitKey(1)
    return numpy.hstack((frame0, frame1))   


def detect():
    # cap = cv2.VideoCapture(1)
    BOARD.setLegacyPattern(True)

    objPoints = []
    imgPoints0 = []
    imgPoints1 = []

    cap0 = createCamera(0)
    cap1 = createCamera(1)
    i = 0
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    while i < 8:
        
        
        path0 = f"./img/stereo/{i}_0.png"
        path1 = f"./img/stereo/{i}_1.png"
        if not os.path.exists(path0) or not os.path.exists(path1):
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1:
                print("Failed to grab frame")
                break

        else:
            frame0 = cv2.imread(path0)
            frame1 = cv2.imread(path1)

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
            tempFrame0 = aruco.drawDetectedCornersCharuco(
                frame0.copy(), charucoCorners0, charucoIds0)
            tempFrame1 = aruco.drawDetectedCornersCharuco(
                frame1.copy(), charucoCorners1, charucoIds1)
        else:
            tempFrame0 = frame0.copy()
            tempFrame1 = frame1.copy()
        finalFrame = combineFrame(tempFrame0, tempFrame1)
        cv2.resizeWindow("Camera Feed", int(finalFrame.shape[1]*0.7), int(finalFrame.shape[0]*0.7))
        cv2.imshow("Camera Feed", finalFrame) 
        if charucoIds0 is not None and charucoIds1 is not None and charucoIds0.size > 4 and charucoIds1.size > 4 and charucoIds0.size == charucoIds1.size:
            temp1, temp2 = BOARD.matchImagePoints(
                charucoCorners0, charucoIds0)
            _, temp3 = BOARD.matchImagePoints(charucoCorners1, charucoIds1)
            print(f"Found {len(temp1)} charuco corners on image  {i}_0")
            if cv2.waitKey(1) & 0xFF == ord('s'):
                objPoints.append(temp1)
                imgPoints0.append(temp2)
                imgPoints1.append(temp3)
                i += 1
                print(f"Saved image pair {i}")
                #savePicture(frame0, frame1, i)
            sleep(1)  # Wait for half a second before capturing the next frame
            
        else:
            print("No charuco corners found on image pair " + str(i))

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
    flags = cv2.CALIB_FIX_INTRINSIC
    R = numpy.zeros((3, 3), dtype=numpy.float64)
    T = numpy.zeros((3, 1), dtype=numpy.float64)
    E = numpy.zeros((3, 3), dtype=numpy.float64)
    F = numpy.zeros((3, 3), dtype=numpy.float64)
    ret = cv2.stereoCalibrate(objPoints, imgPoints0, imgPoint1, cameraMatrix0, distCoeffs0,
                              cameraMatrix1, distCoeffs1, size,  R, T, E, F, flags=flags)
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


def StereoCameraCalibration():
    # if not os.path.exists(f"./img/stereo/0_1.png"):
    # takePicture()
    objPoints, imgPoints0, imgPoints1, size = detect()
    return calibrate(objPoints, imgPoints0, imgPoints1, size)


def main():

    StereoCameraCalibration()


if __name__ == "__main__":
    main()
