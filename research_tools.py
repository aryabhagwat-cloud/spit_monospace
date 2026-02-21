import numpy as np
import cv2

class PerceptionEvaluator:
    def __init__(self, CLASS_DEFS):
        self.classes = CLASS_DEFS

    def calculate_robustness_drop(self, baseline_mask, perturbed_mask):
        """Quantifies how much safety degrades under dust/noise"""
        # Calculate safety for both
        b_score = self.get_safety_score(baseline_mask)
        p_score = self.get_safety_score(perturbed_mask)
        return round(b_score - p_score, 4)

    def get_safety_score(self, mask):
        """Standardized safety score calculation"""
        total_px = mask.size
        lethal_px = np.isin(mask, [2, 3, 4, 5]).sum()
        ratio = lethal_px / (total_px + 1e-6)
        return max(0, 100 - (ratio * 300))

    def get_navigation_complexity(self, mask):
        """Planner-based complexity: Measures obstacle clustering"""
        # Create binary mask of lethal obstacles
        lethal_mask = np.isin(mask, [2, 3, 4, 5]).astype(np.uint8) * 255
        num_labels, _ = cv2.connectedComponents(lethal_mask)
        # More clusters = more planning 'effort' required
        return num_labels