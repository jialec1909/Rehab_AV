import cv2
from tracker import HandTracker
import numpy
import os
import json
import matplotlib.pyplot as plt
from plot import plot_points
from preset_recorder import record_demo_preset, load_demo_preset
# define fixed camera parameters


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

def generate_elliptical_preset(center_x=640, center_y=360, a=400, b=250, steps=400, freq=0.05):
    return [(center_x + a * numpy.sin(i * freq), center_y + b * numpy.cos(i * freq)) for i in range(steps)]

def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = numpy.array(A).reshape((4,4))
    #print('A: ')
    #print(A)
 
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def main():
    cap0 = cv2.VideoCapture(0)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1 = cv2.VideoCapture(1)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    # 2 trackers, if there is one then it will be used for both cameras, which then gave camera 1 bogus data
    # predefine the trajectory as guidance
    #preset = [(200 + 50 * numpy.sin(i * 0.1), 300 + 30 * numpy.cos(i * 0.1)) for i in range(512)]
    #preset = generate_elliptical_preset()


    # create 2 trackers, one for each camera
    tracker0 = HandTracker(preset_trajectory=None, tolerance=40)
    tracker1 = HandTracker(preset_trajectory=None, tolerance=40)

    record_demo = True
    demo_name = "demo_preset.json"
    if not os.path.exists("./demo"):
        os.makedirs("./demo")
    demo_path = f"./demo/{demo_name}"
    # if demo preset exists, load it, otherwise record a new one
    if record_demo:
        print("Recording demo preset...")
        preset_0, preset_1 = record_demo_preset(cap0, tracker0, cap1, tracker1, demo_path=demo_path)

    preset_0, preset_1 = load_demo_preset(demo_path)
    # if demo_preset is None:
    #     demo_preset = record_demo_preset(cap0, tracker0, demo_path="./demo/demo_preset.json")
    # predefine the trajectory as guidance
    tracker0.preset_trajectory = preset_0
    tracker1.preset_trajectory = preset_1

    
    fig = plt.figure()
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        frame0, landmark0 = tracker0.process_frame(frame0)
        tracker0.draw_trajectory_smooth(frame0)
        tracker0.draw_preset_trajectory_with_tolerance(frame0)

        frame1, landmark1 = tracker1.process_frame(frame1)
        tracker1.draw_trajectory_smooth(frame1)
        tracker1.draw_preset_trajectory_with_tolerance(frame1)

        points = []
        if landmark0 and landmark1:
            for i in range(21):
                point0 = numpy.array([landmark0.landmark[i].x, landmark0.landmark[i].y])
                point1 = numpy.array([landmark1.landmark[i].x, landmark1.landmark[i].y])
                # triangulate points
                points_3d = triangulates(point0, point1)
                points.append([i, points_3d[0], points_3d[1], points_3d[2]])
                
        cv2.imshow("Rehab Tracker 0", frame0)
        cv2.imshow("Rehab Tracker 1", frame1)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            plot_points(fig, points)
 
        

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def triangulates(point0, point1):
    
    projectionMatrix0 = numpy.concatenate([numpy.eye(3), [[0],[0],[0]]], axis = -1)
    projectionMatrix1 = numpy.concatenate([ROTATION_MATRIX, TRANSLATION_VECTOR], axis = -1)
    
    #points_4d = cv2.triangulatePoints(projectionMatrix0, projectionMatrix1, point0, point1)
    # Convert from homogeneous coordinates to 3D
    #points_3d = points_4d[:3] / points_4d[3]
    points_3d = DLT(projectionMatrix0, projectionMatrix1, point0, point1)
    return points_3d

if __name__ == "__main__":
    loadCameraParams()
    main()


# point transformation idea
# hand landmarks return x and y in [0, 1] range, so we need to scale them to the image size
# slave camera 1 coordinates to camera 0
# but what to do with z?
# from doc: The z coordinate represents the landmark depth, with the depth at the wrist being the origin
# so all z coord are larger than 0, and the wrist is at z=0, no upper bound
# technically we can use the wrist z as the anchor point, but it is necessary?
# 2 cameras so we can just triangulate the point in 3D space, no z needed
# trangulate should look something like this:
# step 1: convert all the x y to camera coordinates
# step 2: convert points from cam 1 to coordinates of cam 0
# for this step opencv have triangulatePoints(), which should be good enough
# step 3: visualize
