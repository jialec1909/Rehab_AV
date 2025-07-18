from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import enum

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
    
def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.array(points)[:,1], np.array(points)[:,2], np.array(points)[:,3]
    ax.scatter(x, y, z, c='r', marker='o')

    # Define hand connections (pairs of indices)
    connections = [
        (0,1),(1,2),(2,3),(3,4),         # Thumb
        (0,5),(5,6),(6,7),(7,8),         # Index
        (0,9),(9,10),(10,11),(11,12),    # Middle
        (0,13),(13,14),(14,15),(15,16),  # Ring
        (0,17),(17,18),(18,19),(19,20),  # Pinky
        (5,9), (9,13), (13,17),  # Index to Middle to Ring to Pinky
    ]

    # Draw lines for each connection
    for start, end in connections:
        pt_start = points[start]
        pt_end = points[end]
        ax.plot([pt_start[1], pt_end[1]], [pt_start[2], pt_end[2]], [pt_start[3], pt_end[3]], c='lime', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Skeleton')
    plt.draw()
    plt.pause(0.1)  # Pause to allow the plot to update
    
#
#with open("point.txt", "r") as f:
#    points = []
#    for line in f:
#        #match = re.match(r"Point\s+(\d+):\s+\(np\.float64\(([^)]+)\), np\.float64\(([^)]+)\), np\.float64\(([^)]+)\)\)",line)
#        match = re.match(r"Point\s+(\d+):\s+\(array\(\[([^\]]+)\]\), array\(\[([^\]]+)\]\), array\(\[([^\]]+)\]\)\)", line)
#        if match:
#            idx = int(match.group(1))
#            x = float(match.group(2))
#            y = float(match.group(3))
#            z = float(match.group(4))
#            points.append([idx, x, y, z])
#        if len(points) == 21:
#            plot_points(points)
#            points = []
#
