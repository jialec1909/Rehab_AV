import numpy as np
import os
import matplotlib.pyplot as plt
import enum
import json
import math
from matplotlib.widgets import Slider


class HandLandmark(enum.Enum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


def plot_points(ax, points):
    x, y, z = np.array(points)[:, 1], np.array(
        points)[:, 2], np.array(points)[:, 3]
    ax.scatter(x, y, z, c='r', marker='o')

    # Define hand connections (pairs of indices)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Index to Middle to Ring to Pinky
    ]

    # Draw lines for each connection
    for start, end in connections:
        pt_start = points[start]
        pt_end = points[end]
        ax.plot([pt_start[1], pt_end[1]], [pt_start[2], pt_end[2]],
                [pt_start[3], pt_end[3]], c='lime', linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([0.1, 0.3])
    ax.set_ylim([0.1, 0.3])
    ax.set_zlim([0.3, 0.5])
    ax.set_title('3D Hand Skeleton')
    dist= math.sqrt(math.pow(x[0]-x[9], 2) +
                   math.pow(y[0]-y[9], 2) +
                   math.pow(z[0]-z[9], 2))
    print(f"Distance between wrist and middle finger: {dist:.20f} units")


def plotFromFile():
    plt.close('all')
    path = "./img/handPosition.json"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return

    with open(path, "r") as file:
        data = file.read()
    listOfHandLandmarks = json.loads(data)
    print(f"Loaded {len(data)} frames of hand landmarks.")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)

    # Slider setup
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(
        listOfHandLandmarks)-1, valinit=0, valstep=1)

    def update(val):
        ax.cla()  # Clear the current axes
        frame = int(slider.val)
        plot_points(ax, listOfHandLandmarks[frame])
        plt.draw()
        

    slider.on_changed(update)
    plt.show()


def plotFromData(queue):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    while True:
        points = queue.get()
        if points is None:
            break
        ax.cla()
        plot_points(ax, points)
        plt.pause(0.001)
    plt.close(fig)
    
