import subprocess
import re

def get_avfoundation_cameras():
    try:
        result = subprocess.run(
            ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '""'],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

        output = result.stderr

        video_devices = []
        in_video_section = False

        for line in output.splitlines():
            if "AVFoundation video devices:" in line:
                in_video_section = True
                continue
            if "AVFoundation audio devices:" in line:
                in_video_section = False
            if in_video_section and re.search(r"\[\d+\]", line):
                match = re.search(r"\[(\d+)\] (.+)", line)
                if match:
                    index = int(match.group(1))
                    name = match.group(2).strip()
                    video_devices.append((index, name))
        return video_devices
    except Exception as e:
        print("Error detecting cameras:", e)
        return []

def find_camera_indices():
    cameras = get_avfoundation_cameras()
    cam_indices = {}

    for index, name in cameras:
        name_lower = name.lower()
        if "iriun" in name_lower:
            cam_indices["iriun"] = index
        elif "facetime" in name_lower:
            cam_indices["facetime"] = index

    print("Detected video devices:")
    for index, name in cameras:
        print(f"  Camera {index}: {name}")

    return cam_indices

if __name__ == "__main__":
    cam_map = find_camera_indices()
    if "iriun" in cam_map and "facetime" in cam_map:
        print(f"Iriun: Camera {cam_map['iriun']}, FaceTime: Camera {cam_map['facetime']}")
    elif "iriun" in cam_map:
        print(f"Only Iriun detected: Camera {cam_map['iriun']}")
    elif "facetime" in cam_map:
        print(f"Only FaceTime detected: Camera {cam_map['facetime']}")
    else:
        print("No usable camera detected.")
