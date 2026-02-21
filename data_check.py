import cv2
import numpy as np
import os
import pandas as pd

# Path Configuration (Update these to your local paths)
MASK_FOLDER = "path/to/your/masks"
REPORT_NAME = "dataset_health_report.csv"

def run_audit():
    results = []
    print(f"🔍 Auditing Dataset...")

    if not os.path.exists(MASK_FOLDER):
        print(f"❌ Error: Folder {MASK_FOLDER} not found.")
        return

    for filename in os.listdir(MASK_FOLDER):
        if filename.endswith(('.png', '.jpg')):
            path = os.path.join(MASK_FOLDER, filename)
            mask = cv2.imread(path, 0)
            
            # Imbalance Ratio: (Rocks + Logs + Trees + Bushes) / Total
            lethal_pixels = np.isin(mask, [2, 3, 4, 5]).sum()
            imb_ratio = lethal_pixels / mask.size
            
            # Hard-Case Mining: Flagging technical terrain
            is_hard = "Yes" if imb_ratio > 0.12 else "No"
            
            results.append({
                "Filename": filename,
                "Imbalance_Ratio": round(imb_ratio, 4),
                "Hard_Case": is_hard
            })

    df = pd.DataFrame(results)
    df.to_csv(REPORT_NAME, index=False)
    print(f"✅ Success! Report saved to {REPORT_NAME}")
    print(df['Hard_Case'].value_counts(normalize=True) * 100)

if __name__ == "__main__":
    run_audit()