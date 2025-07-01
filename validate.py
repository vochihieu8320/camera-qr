import cv2

def check_available_cameras():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        print(f"Camera {index} is available.")
        cap.release()
        index += 1

    if index == 0:
        print("No cameras found.")
    else:
        print(f"Total available cameras: {index}")

check_available_cameras()
