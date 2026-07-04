import cv2
from core.detection.capture import get_frame

def run():
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        frame = get_frame(camera)
        if frame is None:
            break   

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()   