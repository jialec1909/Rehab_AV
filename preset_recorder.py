import cv2
import json
import mediapipe as mp
import os


def record_demo_preset(cap, tracker, demo_path="demo_preset.json"):
    """
    Interactive recording of a demo trajectory using camera `cap` and `tracker`.

    Controls shown on-screen:
      - 'd' or 'D': start/stop recording demo
      - 'q' or 'Q' or ESC: finish and save demo

    Displays live landmarks + recording status, and saves result to JSON.
    Returns list of (x, y) points.
    """
    recording = False
    demo_buffer = []
    window_name = "Record Demo Preset"
    # Prepare window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print(f"[Preset Recorder] Save path: {os.path.abspath(demo_path)}")
    print("[Preset Recorder] Press 'd' to start/stop recording, 'q'/ESC to finish and save.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Preset Recorder] Camera disconnected.")
            break

        disp = frame.copy()
        # Draw instructions on-frame
        cv2.putText(disp, "d: rec/stop | q or ESC: save & exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Process landmarks
        disp, lm = tracker.process_frame(disp)

        # If recording, collect points and display count
        if recording and lm:
            h, w, _ = frame.shape
            tip = lm.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            demo_buffer.append((tip.x * w, tip.y * h))
            cv2.putText(disp, f"Rec: {len(demo_buffer)} pts", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, disp)
        # Use slightly longer wait time for key capture reliability
        key = cv2.waitKey(30)
        if key != -1:
            # Mask off any non-ascii bits
            k = key & 0xFF
            # Print for debug
            print(f"[Preset Recorder] Key pressed: {k}")
            # Toggle recording on 'd' or 'D'
            if k in (ord('d'), ord('D')):
                recording = not recording
                state = "started" if recording else "stopped"
                print(f"[Preset Recorder] Recording {state}. Total points: {len(demo_buffer)}")
            # Exit on 'q', 'Q', or ESC (27)
            elif k in (ord('q'), ord('Q'), 27):
                print("[Preset Recorder] Finishing and saving...")
                break

    cv2.destroyWindow(window_name)

    # Ensure directory exists
    dir_path = os.path.dirname(demo_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save to JSON
    try:
        with open(demo_path, 'w') as f:
            json.dump(demo_buffer, f)
        print(f"[Preset Recorder] Saved demo preset ({len(demo_buffer)} pts) to: {os.path.abspath(demo_path)}")
    except Exception as e:
        print(f"[Preset Recorder] Failed to save demo preset: {e}")

    return demo_buffer


def load_demo_preset(demo_path="./demo/demo_preset.json"):
    """
    Load a previously saved demo preset from JSON. Returns point list or None.
    """
    try:
        with open(demo_path, 'r') as f:
            pts = json.load(f)
        print(f"[Preset Recorder] Loaded demo preset ({len(pts)} pts) from: {os.path.abspath(demo_path)}")
        return pts
    except Exception as e:
        print(f"[Preset Recorder] No demo preset found at {os.path.abspath(demo_path)}: {e}")
        return None
