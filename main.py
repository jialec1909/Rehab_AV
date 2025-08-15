import cv2
from tracker import HandTracker
import numpy
import os
import json
import plot
import multiprocessing as mp
from helper import createCamera, combineFrame, triangulates, loadCameraParams
#from preset_recorder import record_demo_preset, load_demo_preset, record_actual_movement, load_actual_movement
# define fixed camera parameters

def main():
    cap0 = createCamera(0)
    cap1 = createCamera(1)
    tracker0 = HandTracker(preset_trajectory=None, tolerance=40)
    tracker1 = HandTracker(preset_trajectory=None, tolerance=40)
    
    queue = mp.Queue()
    process = mp.Process(target=plot.plotFromLive, args=(queue,))
    process.start()
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    listOfHandLandmarks = []
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        frame0, landmark0 = tracker0.process_frame(frame0)
        #tracker0.draw_trajectory_smooth(frame0)
        #tracker0.draw_preset_trajectory_with_tolerance(frame0)

        frame1, landmark1 = tracker1.process_frame(frame1)
        #tracker1.draw_trajectory_smooth(frame1)
        #tracker1.draw_preset_trajectory_with_tolerance(frame1)

        finalFrame = combineFrame(frame0, frame1)
        cv2.resizeWindow("Camera Feed", int(finalFrame.shape[1]*0.7), int(finalFrame.shape[0]*0.7))
        cv2.imshow("Camera Feed", finalFrame) 
        
        if landmark0 and landmark1:
            points = []
            for i in range(21):
                point0 = numpy.array([landmark0.landmark[i].x, landmark0.landmark[i].y])
                point1 = numpy.array([landmark1.landmark[i].x, landmark1.landmark[i].y])
                # triangulate points
                points_3d = triangulates(point0, point1)
                points.append([i, points_3d[0], points_3d[1], points_3d[2]])
            listOfHandLandmarks.append(points)
            if queue.qsize() <= 0:
                queue.put(points)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            with open("./img/handPosition.json", "w") as output:
                json.dump(listOfHandLandmarks, output, indent=4)    
            process.terminate()
            break
        
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    loadCameraParams()
    main()
    #plot.plotFromFile()

