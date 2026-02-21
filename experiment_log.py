import pandas as pd
import datetime

# --- PERSON 3: EXPERIMENT TRACKING SYSTEM ---

def log_experiment(version, architecture, loss_func, miou, stability_index):
    """Tracks model versions to find the best optimized core"""
    
    new_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Version": version,
        "Architecture": architecture,
        "Loss_Function": loss_func,
        "mIoU": round(miou, 4),
        "Stability": round(stability_index, 2)
    }
    
    try:
        df = pd.read_csv("model_experiments.csv")
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([new_entry])
        
    df.to_csv("model_experiments.csv", index=False)
    print(f"🚀 Experiment {version} Logged Successfully!")

# Example: Testing a stronger backbone (Responsibility 4)
# log_experiment("v1.2", "DeepLabV3+ (ResNet50)", "Dice + Focal", 0.782, 0.85)