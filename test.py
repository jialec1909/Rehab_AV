import cv2
from tracker import HandTracker
import numpy
import os
import json
import plot
import multiprocessing as mp
from helper import createCamera, combineFrame, resizeFrame, triangulates, loadCameraParams, setBoardParameters
from cv2 import aruco

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
    global BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS 
    BOARD, CHARUCO_PARAMS, DETECTOR_PARAMS = setBoardParameters()
    loadCameraParams()
    main()
    #plot.plotFromFile()

