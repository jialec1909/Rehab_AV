import cv2
import json
import mediapipe as mp
import os
import numpy as np  


def record_demo_preset(cap0, tracker0, cap1, tracker1, demo_path="demo_preset.json"):
    """
    Interactive stereo demo-preset recording.

    Controls in window:
        d/D: start/stop recording
        q/Q/ESC: finish & save

    Overlays live landmarks and trails in both camera views,
    stores pairs of points [x0,y0,x1,y1] internally,
    then splits into two presets for cam0/cam1.

    Saves JSON with structure:
      {"cam0": [[x0,y0], ...], "cam1": [[x1,y1], ...]}

    Returns:
      preset0: list of (x0, y0)
      preset1: list of (x1, y1)
    """
    recording = False
    demo_buffer = []  # each entry [x0,y0,x1,y1]

    win0, win1 = "Record Demo Cam0", "Record Demo Cam1"
    for w in (win0, win1):
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, 640, 480)

    print(f"[Preset Recorder] Will save to: {os.path.abspath(demo_path)}")
    print("[Preset Recorder] Press 'd' to start/stop, 'q'/ESC to finish and save.")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("[Preset Recorder] Camera disconnected.")
            break

        disp0, disp1 = frame0.copy(), frame1.copy()
        instruct = "d:rec/stop | q/ESC:save&exit"
        cv2.putText(disp0, instruct, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(disp1, instruct, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

        disp0, lm0 = tracker0.process_frame(disp0)
        disp1, lm1 = tracker1.process_frame(disp1)

        # draw recorded trajectory so far
        for x0,y0,x1,y1 in demo_buffer:
            cv2.circle(disp0, (int(x0),int(y0)), 3, (0,200,0), -1)
            cv2.circle(disp1, (int(x1),int(y1)), 3, (0,200,0), -1)

        # record current if enabled and both visible
        if recording and lm0 and lm1:
            h0,w0,_ = frame0.shape
            h1,w1,_ = frame1.shape
            tip0 = lm0.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            tip1 = lm1.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            x0, y0 = tip0.x * w0, tip0.y * h0
            x1, y1 = tip1.x * w1, tip1.y * h1
            demo_buffer.append([x0, y0, x1, y1])
            cv2.putText(disp0, f"Rec pts: {len(demo_buffer)}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.putText(disp1, f"Rec pts: {len(demo_buffer)}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow(win0, disp0)
        cv2.imshow(win1, disp1)

        key = cv2.waitKey(30)
        if key != -1:
            k = key & 0xFF
            if k in (ord('d'), ord('D')):
                recording = not recording
                state = "started" if recording else "stopped"
                print(f"[Preset Recorder] Recording {state}. Points so far: {len(demo_buffer)}")
            elif k in (ord('q'), ord('Q'), 27):
                print("[Preset Recorder] Ending recording...")
                break

    cv2.destroyWindow(win0)
    cv2.destroyWindow(win1)

    # split buffers
    preset0 = [(pt[0], pt[1]) for pt in demo_buffer]
    preset1 = [(pt[2], pt[3]) for pt in demo_buffer]

    # save JSON
    data = {"cam0": preset0, "cam1": preset1}
    os.makedirs(os.path.dirname(demo_path) or '.', exist_ok=True)
    try:
        with open(demo_path, 'w') as f:
            json.dump(data, f)
        print(f"[Preset Recorder] Saved presets (cam0:{len(preset0)}pts, cam1:{len(preset1)}pts) to {os.path.abspath(demo_path)}")
    except Exception as e:
        print(f"[Preset Recorder] Save error: {e}")

    return preset0, preset1


def load_demo_preset(demo_path="demo_preset.json"):
    """
    Loads a JSON with keys 'cam0','cam1', each list of [x,y].
    Returns (preset0, preset1) or (None,None).
    """
    try:
        with open(demo_path, 'r') as f:
            data = json.load(f)
        preset0 = data.get('cam0')
        preset1 = data.get('cam1')
        if preset0 is None or preset1 is None:
            raise ValueError('Missing cam0 or cam1 keys')
        print(f"[Preset Recorder] Loaded presets (cam0:{len(preset0)}pts, cam1:{len(preset1)}pts)")
        return preset0, preset1
    except Exception as e:
        print(f"[Preset Recorder] Load failed: {e}")
        return None, None


def record_actual_movement(cap0, tracker0, cap1, tracker1,
                           preset0, preset1, tolerance=40,
                           actual_path="./demo/actual_movement.json"):
    """
    Records actual movement only after index finger enters the start point of preset.

    Args:
      cap0, tracker0: camera0 and its tracker
      cap1, tracker1: camera1 and its tracker
      preset0, preset1: lists of (x,y) start-to-finish preset points for each cam
      tolerance: pixel radius around preset0[0]/preset1[0] defining start region
      actual_path: where to save JSON as {"cam0": [...], "cam1": [...]}

    Returns:
      actual0, actual1: lists of recorded (x,y) points after entry
    """
    started = False
    actual_buffer = []  # [x0,y0,x1,y1]

    win0, win1 = "Actual Rec Cam0", "Actual Rec Cam1"
    for w in (win0, win1):
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, 640, 480)

    print(f"[Actual Recorder] Waiting for finger to enter preset start region (tol={tolerance})...")
    print("[Actual Recorder] Press ESC or 'q' to finish recording once started.")

    start0 = np.array(preset0[0])
    start1 = np.array(preset1[0])

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("[Actual Recorder] Camera disconnected.")
            break

        disp0, disp1 = frame0.copy(), frame1.copy()
        disp0, lm0 = tracker0.process_frame(disp0)
        disp1, lm1 = tracker1.process_frame(disp1)

        # draw start circles
        cv2.circle(disp0, (int(start0[0]),int(start0[1])), tolerance, (255,0,0),2)
        cv2.circle(disp1, (int(start1[0]),int(start1[1])), tolerance, (255,0,0),2)

        # If started, overlay past
        if started:
            for x0,y0,x1,y1 in actual_buffer:
                cv2.circle(disp0,(int(x0),int(y0)),2,(0,200,0),-1)
                cv2.circle(disp1,(int(x1),int(y1)),2,(0,200,0),-1)

        # check entry and record
        if lm0 and lm1:
            tip0 = lm0.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            tip1 = lm1.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            h0,w0,_ = frame0.shape
            h1,w1,_ = frame1.shape
            pt0 = np.array([tip0.x*w0, tip0.y*h0])
            pt1 = np.array([tip1.x*w1, tip1.y*h1])

            if not started:
                # auto-start when both tips within tolerance of start
                if np.linalg.norm(pt0-start0)<=tolerance and np.linalg.norm(pt1-start1)<=tolerance:
                    started = True
                    print("[Actual Recorder] Movement entry detected. Start recording.")
            else:
                # record points
                actual_buffer.append([float(pt0[0]), float(pt0[1]), float(pt1[0]), float(pt1[1])])
                cv2.putText(disp0, f"Rec pts: {len(actual_buffer)}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                cv2.putText(disp1, f"Rec pts: {len(actual_buffer)}", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow(win0, disp0)
        cv2.imshow(win1, disp1)
        key = cv2.waitKey(30) & 0xFF
        if started and key in (27, ord('q'), ord('Q')):
            print("[Actual Recorder] Stopping recording.")
            break

    cv2.destroyWindow(win0)
    cv2.destroyWindow(win1)

    # split and save
    actual0 = [(pt[0],pt[1]) for pt in actual_buffer]
    actual1 = [(pt[2],pt[3]) for pt in actual_buffer]
    data = {"cam0": actual0, "cam1": actual1}
    os.makedirs(os.path.dirname(actual_path) or '.', exist_ok=True)
    try:
        with open(actual_path,'w') as f:
            json.dump(data,f)
        print(f"[Actual Recorder] Saved actual movement ({len(actual0)} pts) to {os.path.abspath(actual_path)}")
    except Exception as e:
        print(f"[Actual Recorder] Save error: {e}")

    return actual0, actual1


def load_actual_movement(actual_path="./demo/actual_movement.json"):
    try:
        with open(actual_path, 'r') as f:
            data = json.load(f)
        actual0 = data.get("cam0")
        actual1 = data.get("cam1")
        if actual0 is None or actual1 is None:
            raise KeyError("Missing 'cam0' or 'cam1'")
        print(f"[Actual Loader] Loaded actual movement (cam0:{len(actual0)} pts, cam1:{len(actual1)} pts)")
        return actual0, actual1
    except Exception as e:
        print(f"[Actual Loader] Load failed: {e}")
        return None, None