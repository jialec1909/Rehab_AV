import cv2
import numpy

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