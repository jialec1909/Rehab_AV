import mediapipe as mp
import numpy as np
import cv2
from collections import deque

class HandTracker:
    def __init__(self, max_len=512, preset_trajectory=None, tolerance=1000):
        self.hands = mp.solutions.hands.Hands()
        self.trajectory = deque(maxlen=max_len)  # store finger trajectory
        self.mp_draw = mp.solutions.drawing_utils
        self.preset_trajectory = preset_trajectory  # list of (x, y)
        self.tolerance = tolerance  # in pixels

    def check_current_point_status(self):
        if not self.preset_trajectory or len(self.trajectory) == 0:
            return None

        current_pt = np.array(self.trajectory[-1])

        dists = [np.linalg.norm(current_pt - np.array(ref_pt)) for ref_pt in self.preset_trajectory]
        min_dist = min(dists)

        return min_dist <= self.tolerance

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        height, width, _ = frame.shape

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # index finger tip
                tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(tip.x * width), int(tip.y * height)
                self.trajectory.append((x, y))
                #self.trajectory.append((x, y, tip.z))

                self.mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            return frame, hand_landmarks
        else:
            # If no hands detected, return the original frame
            return frame, None

    def draw_trajectory(self, frame):
        for i in range(1, len(self.trajectory)):
            cv2.line(frame, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 2)

    def draw_preset_trajectory(self, frame):
        if not self.preset_trajectory:
            return

        for pt in self.preset_trajectory:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)


    
    def draw_trajectory_smooth(self, frame):
        overlay = frame.copy()
        smoothed = self.smooth_trajectory(self.trajectory)
        #for i in range(1, len(smoothed)):
            #cv2.line(frame, smoothed[i - 1], smoothed[i], (0, 200, 0), 2)
        N = len(smoothed)

        for i in range(1, N):
            pt1 = smoothed[i - 1]
            pt2 = smoothed[i]

            if self.preset_trajectory:
                dists = [np.linalg.norm(np.array(pt2) - np.array(p)) for p in self.preset_trajectory]
                safe = min(dists) <= self.tolerance
            else:
                safe = True

            alpha = i / N
            color = (0, int(120 + 135 * alpha), 0) if safe else (0, 0, int(120 + 135 * alpha))
            thickness = int(1 + 3 * alpha)

            cv2.line(overlay, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        if N > 0:
            last_pt = smoothed[-1]
            if self.preset_trajectory:
                dists = [np.linalg.norm(np.array(last_pt) - np.array(p)) for p in self.preset_trajectory]
                safe = min(dists) <= self.tolerance
            else:
                safe = True

            dot_color = (0, 255, 0) if safe else (0, 0, 255)
            cv2.circle(frame, last_pt, 8, dot_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, last_pt, 12, dot_color, 2, lineType=cv2.LINE_AA)        
            # cv2.circle(frame, smoothed[-1], 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            # cv2.circle(frame, smoothed[-1], 12, (0, 255, 0), 2, lineType=cv2.LINE_AA)

    def draw_trajectory_pseudo3D(self, frame):
        for i in range(1, len(self.trajectory)):
            x1, y1, z1 = self.trajectory[i - 1]
            x2, y2, z2 = self.trajectory[i]

            norm_z = max(min(-(z1 + z2) / 2 * 5, 1.0), 0.0)
            thickness = int(1 + 4 * norm_z)
            color_intensity = int(255 * norm_z)

            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, color_intensity, 0),
                thickness
            )

    def smooth_trajectory(self, trajectory, window_size=5):
        trajectory = list(trajectory)
        if len(trajectory) < window_size:
            return list(trajectory)
        smoothed = []
        for i in range(len(trajectory)):
            start = max(0, i - window_size + 1)
            window = trajectory[start:i+1]
            xs = [pt[0] for pt in window]
            ys = [pt[1] for pt in window]
            smoothed.append((int(np.mean(xs)), int(np.mean(ys))))
        return smoothed
    
    def draw_preset_trajectory_with_tolerance(self, frame):

        if not self.preset_trajectory:
            return

        overlay = frame.copy()

        for pt in self.preset_trajectory:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(overlay, (x, y), self.tolerance, (100, 200, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x, y), 2, (200, 220, 255), -1, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
