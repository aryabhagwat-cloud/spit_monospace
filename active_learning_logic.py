import os
import shutil
import pandas as pd

def harvest_hard_cases(audit_csv, output_folder="data_factory/to_be_labeled"):
    # Load the audit results we just generated
    df = pd.read_csv(audit_csv)
    os.makedirs(output_folder, exist_ok=True)
    
    # Identify images where the prediction found 0% obstacles but SHOULD have seen some
    # We focus on the "pred_" files with 0.00% obstacle area
    hard_cases = df[(df['filename'].str.contains('pred')) & (df['Obstacle_Area%'] == 0.00)]
    
    for _, row in hard_cases.iterrows():
        source = os.path.join("masks", row['filename'])
        if os.path.exists(source):
            shutil.copy(source, os.path.join(output_folder, row['filename']))
            print(f"🚩 Edge Case Saved: {row['filename']} (Model missed all obstacles)")

    print(f"\n✅ Business MVO: {len(hard_cases)} failure cases moved to '{output_folder}' for retraining.")

harvest_hard_cases("dataset_audit_v2.csv")