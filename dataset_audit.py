import cv2
import numpy as np
import os
import pandas as pd

# Define your classes based on the colors in your masks
CLASS_MAP = {
    0: "Sand",
    2: "Rocks",
    5: "Bushes"
}

def audit_dataset(mask_dir):
    stats = []
    for file in os.listdir(mask_dir):
        if file.startswith("pred_") or not file.endswith(".png"):
            continue
            
        mask = cv2.imread(os.path.join(mask_dir, file), 0)
        h, w = mask.shape
        total_pixels = h * w
        
        file_stats = {"filename": file}
        for val, name in CLASS_MAP.items():
            count = np.sum(mask == val)
            file_stats[name] = round((count / total_pixels) * 100, 2)
        
        stats.append(file_stats)
    
    return pd.DataFrame(stats)

df = audit_dataset("masks")
print("--- Dataset Class Distribution (%) ---")
print(df)
df.to_csv("dataset_audit.csv", index=False)