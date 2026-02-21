import cv2
import numpy as np
import os

mask_path = "masks/image1.png" # Pick one image that should have data
mask = cv2.imread(mask_path, 0)

if mask is not None:
    unique_values = np.unique(mask)
    print(f"✅ The unique pixel values found in your mask are: {unique_values}")
else:
    print("❌ Could not find the mask file. Check your file path!")