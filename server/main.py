import cv2
from core.detection.capture import get_frame
from core.detection.model import initialize_model, detect_objects, get_results

def run():
    """
    Main function to run the object detection on camera feed.
    """
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        frame = get_frame(camera)
        if frame is None:
            break   

        model = initialize_model()
        results = detect_objects(model, frame)
        image = results[0].plot()
        cv2.imshow('Camera Feed', image)

        results_data = get_results(model, frame)
        print(results_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()   