from config import CONFIDENCE_THRESHOLD

def evaluate(detection):
    confidence = detection.get("confidence", 0) if detection else 0
    return confidence >= CONFIDENCE_THRESHOLD
