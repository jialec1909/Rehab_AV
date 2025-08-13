import cv2
import numpy
import os
import json
from cv2 import aruco

def setBoardParameters():
    #BOARD = aruco.CharucoBoard((7, 4), 0.025, 0.018, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
    BOARD = aruco.CharucoBoard((8, 6), 0.03, 0.022, aruco.getPredefinedDictionary(aruco.DICT_4X4_1000))
    CHARUCO_PARAMS = aruco.CharucoParameters()
    DETECTOR_PARAMS = aruco.DetectorParameters()
    return BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS

def createCamera(camID):
    cap = cv2.VideoCapture(camID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap

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
    
def combineFrame(frame0, frame1):
    if frame0.shape[0] >= frame1.shape[0]:
        newWidth = int(frame0.shape[1] / frame0.shape[0] * frame1.shape[0])
        frame0 = cv2.resize(frame0, (newWidth, frame1.shape[0]))
    else:
        newWidth = int(frame1.shape[1] / frame1.shape[0] * frame0.shape[0])
        frame1 = cv2.resize(frame1, (newWidth, frame0.shape[0]))
    cv2.waitKey(1)
    return numpy.hstack((frame0, frame1))

def resizeFrame(frame):
    #1920x1080 is the screen resolution
    if frame.shape[1] != 1920 or frame.shape[0] != 1080:
        scale = min(1920 / frame.shape[1], 1080 / frame.shape[0])
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        return cv2.resize(frame, new_size)
    else :
        return frame

def matchingCorner(corners0, ids0, corners1, ids1):
    matchedID = numpy.intersect1d(ids0, ids1).reshape(-1, 1)
    if len(matchedID) > 0:
        matchedCorners0 = numpy.array(
            [corners0[i] for i in range(len(corners0)) if ids0[i] in matchedID])
        matchedCorners1 = numpy.array(
            [corners1[i] for i in range(len(corners1)) if ids1[i] in matchedID])
    return matchedID, matchedCorners0, matchedCorners1

def triangulates(point0, point1):
    projectionMatrix0 = numpy.concatenate([numpy.eye(3), [[0],[0],[0]]], axis = -1)
    projectionMatrix1 = numpy.concatenate([ROTATION_MATRIX, TRANSLATION_VECTOR], axis = -1)
    points_4d = cv2.triangulatePoints(projectionMatrix0, projectionMatrix1, point0, point1)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
    return points_3d.flatten()