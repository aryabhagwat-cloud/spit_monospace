import cv2
import os
import numpy as np

def finalize_data_integrity(input_folder="masks", output_folder="refined_masks"):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(".png") and not file.startswith("overlay"):
            img = cv2.imread(os.path.join(input_folder, file), 0)
            # THRESHOLDING: High-precision cutoff at 127
            _, clean_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(output_folder, file), clean_mask)
    print("✅ Accuracy Stage 1: Masks refined to Binary (0/255).")

finalize_data_integrity()