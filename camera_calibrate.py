import cv2
import numpy
from cv2 import aruco
import os
from time import sleep
import json
from camera_selector import find_camera_indices

# BOARD = aruco.CharucoBoard((5, 5), 0.015, 0.011, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((6, 8), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
# BOARD = aruco.CharucoBoard((7, 3), 0.030, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
BOARD = aruco.CharucoBoard(
    (8, 6), 0.03, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
)
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()

def list_available_cameras(max_cams=5):
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available")
        cap.release()

def detect(camID):
    # cap = cv2.VideoCapture(1)
    BOARD.setLegacyPattern(True)

    objPoints = []
    imgPoints = []

    # get count of pictures in /img
    imgCount = len(os.listdir(f"./img/{camID}/"))
    for i in range(imgCount):
        # frame = BOARD.generateImage((900,900),10,1)
        frame = cv2.imread(f"./img/{camID}/" + str(i) + ".png")
        size = frame.shape
        # ret, frame = cap.read()
        # if not ret:
        #    print("Failed to grab frame")
        #    break

        # Grayscale the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charucoDetector = aruco.CharucoDetector(
            BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS)

        charucoCorners, charucoIds, _, _ = (
            charucoDetector.detectBoard(gray)
        )

        if charucoCorners is not None and charucoIds is not None and charucoIds.size > 4:
            print(f"[INFO] Image {i}: Detected {len(charucoIds)} charuco corners.")
            frame_display = aruco.drawDetectedCornersCharuco(frame.copy(), charucoCorners, charucoIds)
            cv2.imshow("Charuco Detection", frame_display)
            cv2.waitKey(500)

            temp1, temp2 = BOARD.matchImagePoints(charucoCorners, charucoIds)
            objPoints.append(temp1)
            imgPoints.append(temp2)

        else:
            print("No charuco corners found on image " + str(i))

    # cap.release()
    cv2.destroyAllWindows()
    return objPoints, imgPoints, size[:2]


def createCamera(camID):
    cap = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    return cap


def takePicture(camID):
    if not os.path.exists(f"./img/{camID}"):
        os.makedirs(f"./img/{camID}")
    else:
        print(f"Directory ./img/{camID} already exists. Pictures will be overwritten.")
    i = 0
    cap = createCamera(camID)
    
    while i < 25:
        ret, frame = cap.read()
        cv2.imshow("Camera Feed", frame)
        #cv2.resizeWindow("Camera Feed", 1280, 720)
        if not ret:
            print("Failed to grab frame")
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"./img/{camID}/{i}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            i += 1
    cap.release()
    cv2.destroyAllWindows()


def calibrate(objPoints, imgPoints, size, camID):

    cameraMatrix = numpy.zeros((3, 3), dtype=numpy.float64)
    distCoeffs = numpy.zeros((5, 1), dtype=numpy.float64)
    rvecs = []
    tvecs = []
    error = 0.0
    ret = cv2.calibrateCameraExtended(
        objPoints, imgPoints, (size[0], size[1]
                               ), cameraMatrix, distCoeffs, rvecs, tvecs, perViewErrors=error
    )
    print(f"cam {camID}: " + str(ret[0]))

    # write to txt file
    calibration_data = {
        "camera_matrix": cameraMatrix.tolist(),
        "distortion_coefficients": distCoeffs.tolist()
    }
    with open(f"./img/camera_calibration_{camID}.json", "w") as f:
        json.dump(calibration_data, f, indent=4)
    return cameraMatrix, distCoeffs


def singleCameraCalibration(camID):
    if not os.path.exists(f"./img/{camID}/20.png"):
        takePicture(camID)
    allCharucoCorners, allCharucoIds, size = detect(camID)
    return calibrate(allCharucoCorners, allCharucoIds, size, camID)


def main():
    #list_available_cameras()
    #cam_indices = find_camera_indices()
    #print(cam_indices)
    cam1, dist1 = singleCameraCalibration(0)
    cam2, dist2 = singleCameraCalibration(1)

if __name__ == "__main__":
    main()
