import cv2
import time
from core.detection.capture import get_frame
from core.detection.model import initialize_model, detect_objects, get_results
from core.decision.engine import evaluate

def run():
    """
    Main function to run the object detection on camera feed.
    """
    camera = cv2.VideoCapture(0)
    model = initialize_model()

    while camera.isOpened():
        frame = get_frame(camera)
        if frame is None:
            break   
        start_time = time.time()

        results_data = get_results(model, frame)
        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.2f} seconds.")
        if evaluate(results_data):
            print(f"Detected {results_data['class']} with confidence {results_data['confidence']:.2f}")
            cv2.rectangle(frame, (int(results_data['xyxy'][0]), int(results_data['xyxy'][1])),
                          (int(results_data['xyxy'][2]), int(results_data['xyxy'][3])), (0, 255, 0), 2)
       
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()   