import cv2

def get_frame(camera):
    """
    Read a frame from the camera.
    """
    ret, frame = camera.read()
    if not ret:
        raise ValueError("Failed to capture frame from camera.")
    return frame