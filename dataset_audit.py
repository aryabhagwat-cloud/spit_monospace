import cv2
import numpy as np
import os
import pandas as pd

def audit_dataset_v2(mask_dir):
    stats = []
    for file in os.listdir(mask_dir):
        if not file.endswith(".png") or file.startswith("overlay"): continue
            
        mask = cv2.imread(os.path.join(mask_dir, file), 0)
        total_pixels = mask.size
        
        # Define ranges instead of single numbers
        sand = np.sum((mask > 0) & (mask <= 100))
        obstacles = np.sum(mask > 100)
        
        stats.append({
            "filename": file,
            "Sand_Area%": round((sand / total_pixels) * 100, 2),
            "Obstacle_Area%": round((obstacles / total_pixels) * 100, 2)
        })
    
    return pd.DataFrame(stats)

df = audit_dataset_v2("masks")
print(df)