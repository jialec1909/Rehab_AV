import cv2
from tracker import HandTracker
import numpy
import os
import json
import plot
import multiprocessing as mp
from stereo_calibrate import createCamera, combineFrame, resizeFrame
from cv2 import aruco
#from preset_recorder import record_demo_preset, load_demo_preset, record_actual_movement, load_actual_movement
# define fixed camera parameters
BOARD = aruco.CharucoBoard((8, 6), 0.03, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
CHARUCO_PARAMS = aruco.CharucoParameters()
DETECTOR_PARAMS = aruco.DetectorParameters()
BOARD.setLegacyPattern(True)
def loadCameraParams():
    path = f"./img/stereo_calibration.json"
    if not os.path.exists(path):
        print("calibration parameters file not found.")
        exit(1)
    with open(path) as f:
        data = json.load(f)
    global ROTATION_MATRIX, TRANSLATION_VECTOR, ESSENTIAL_MATRIX, FUNDAMENTAL_MATRIX, CAMERA_INTRINSICS_0, CAMERA_INTRINSICS_1, DISTORTION_COEFFICIENTS_0, DISTORTION_COEFFICIENTS_1
    ROTATION_MATRIX = numpy.array(data["rotation_matrix"])
    TRANSLATION_VECTOR = numpy.array(data["translation_vector"])
    ESSENTIAL_MATRIX = numpy.array(data["essential_matrix"])
    FUNDAMENTAL_MATRIX = numpy.array(data["fundamental_matrix"])
    CAMERA_INTRINSICS_0 = numpy.array(data["camera_intrinsics_0"])
    CAMERA_INTRINSICS_1 = numpy.array(data["camera_intrinsics_1"])
    DISTORTION_COEFFICIENTS_0 = numpy.array(data["distortion_coefficients_0"])
    DISTORTION_COEFFICIENTS_1 = numpy.array(data["distortion_coefficients_1"])

def main():
    cap0 = createCamera(0)
    cap1 = createCamera(1)

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        frame0 = resizeFrame(frame0)
        frame1 = resizeFrame(frame1)
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
            #print(f"Detected {len(charucoIds0)} corners in camera 0 and {len(charucoIds1)} corners in camera 1")
            tempFrame0 = aruco.drawDetectedCornersCharuco(frame0.copy(), charucoCorners0, charucoIds0)
            #tempFrame1 = aruco.drawDetectedCornersCharuco(frame1.copy(), charucoCorners1, charucoIds1)
            # Find corners 8 and 26 in both cameras
            getDistance(charucoIds0, charucoIds1, charucoCorners0, charucoCorners1)
        else:
            tempFrame0 = frame0.copy()
            tempFrame1 = frame1.copy()
        finalFrame = combineFrame(tempFrame0, tempFrame1)
        cv2.resizeWindow("Camera Feed", int(finalFrame.shape[1]*0.7), int(finalFrame.shape[0]*0.7))
        cv2.imshow("Camera Feed", finalFrame)

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def triangulates(point0, point1):
    
    projectionMatrix0 = numpy.concatenate([numpy.eye(3), [[0],[0],[0]]], axis = -1)
    projectionMatrix1 = numpy.concatenate([ROTATION_MATRIX, TRANSLATION_VECTOR], axis = -1)
    
    points_4d = cv2.triangulatePoints(projectionMatrix0, projectionMatrix1, point0, point1)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
    return points_3d.flatten()
        
def getDistance(charucoIds0, charucoIds1, charucoCorners0, charucoCorners1):
    ids0_flat = charucoIds0.flatten()
    ids1_flat = charucoIds1.flatten()
    if 8 in ids0_flat and 26 in ids0_flat and 8 in ids1_flat and 26 in ids1_flat:
        idx0_8 = numpy.where(ids0_flat == 8)[0][0]
        idx0_26 = numpy.where(ids0_flat == 26)[0][0]
        idx1_8 = numpy.where(ids1_flat == 8)[0][0]
        idx1_26 = numpy.where(ids1_flat == 26)[0][0]
        pt0_8 = charucoCorners0[idx0_8][0]
        pt0_26 = charucoCorners0[idx0_26][0]
        pt1_8 = charucoCorners1[idx1_8][0]
        pt1_26 = charucoCorners1[idx1_26][0]
        p3d_8 = triangulates(pt0_8, pt1_8)
        p3d_26 = triangulates(pt0_26, pt1_26)
        dist = numpy.linalg.norm(p3d_8 - p3d_26)
        print(f"Distance between corner 8 and 26: {dist:.6f} meters")
if __name__ == "__main__":
    loadCameraParams()
    main()
    #plot.plotFromFile()

