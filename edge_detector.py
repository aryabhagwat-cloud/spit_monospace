import numpy as np

def detect_edge_case(prediction_probs, threshold=0.7):
    """
    If the model's highest confidence is below the threshold, 
    flag it as an edge case.
    """
    max_confidence = np.max(prediction_probs)
    
    if max_confidence < threshold:
        return True, max_confidence
    return False, max_confidence

# Example: If the AI is only 55% sure, it returns (True, 0.55) -> EDGE CASE!