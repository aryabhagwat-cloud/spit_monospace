import numpy as np

def calculate_iou(pd_mask, gt_mask, num_classes=6):
    """Calculates the overlap between prediction and reality"""
    ious = []
    for cls in range(num_classes):
        intersection = np.logical_and(pd_mask == cls, gt_mask == cls).sum()
        union = np.logical_or(pd_mask == cls, gt_mask == cls).sum()
        
        if union == 0:
            ious.append(float('nan'))  # Skip classes not present
        else:
            ious.append(intersection / union)
    
    return ious, np.nanmean(ious)

# This will be your 'Responsibility 3' baseline verification tool