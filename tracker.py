import mediapipe as mp
import numpy as np
import cv2
from collections import deque

class HandTracker:
    def __init__(self, max_len=512):
        self.hands = mp.solutions.hands.Hands()
        self.trajectory = deque(maxlen=max_len)  # store finger trajectory
        self.mp_draw = mp.solutions.drawing_utils

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

        return frame

    def draw_trajectory(self, frame):
        for i in range(1, len(self.trajectory)):
            cv2.line(frame, self.trajectory[i - 1], self.trajectory[i], (0, 255, 0), 2)

    def draw_trajectory_pseudo3D(self, frame):
        for i in range(1, len(self.trajectory)):
            x1, y1, z1 = self.trajectory[i - 1]
            x2, y2, z2 = self.trajectory[i]

            # 将 z 值归一化为线宽和颜色深度的调节因子
            # 注意：MediaPipe 的 z 值通常为负值，越小越靠近相机
            norm_z = max(min(-(z1 + z2) / 2 * 5, 1.0), 0.0)  # 范围控制在 [0, 1]
            thickness = int(1 + 4 * norm_z)  # 线宽范围 1 ~ 5
            color_intensity = int(255 * norm_z)  # 颜色从浅灰到深绿

            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, color_intensity, 0),  # 深浅绿色
                thickness
            )

