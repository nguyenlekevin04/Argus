from ultralytics import YOLO

def initialize_model():
    """
    Initializes the YOLO model for object detection.
    """
    return YOLO("yolov8n.pt")

def detect_objects(model, frame):
    """
    Detects objects in the given frame using the YOLO model.
    """
    return model.predict(frame, device='mps', verbose=False, imgsz=320)

def get_results(model, frame):
    """
    Gets the detection results from the model for the given frame.
    """
    results = detect_objects(model, frame)
    if len(results[0].boxes) == 0:
        return None
    
    best_idx = results[0].boxes.conf.argmax().item()
    return {
        "class": model.names[results[0].boxes.cls[best_idx].item()],
        "confidence": results[0].boxes.conf[best_idx].item(),
        "xyxy": results[0].boxes.xyxy[best_idx].tolist()
    }