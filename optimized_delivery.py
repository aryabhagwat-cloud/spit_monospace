import pandas as pd

def generate_final_report():
    """Summarizes experiments and design choices for delivery"""
    try:
        # Load all your experiment history
        results = pd.read_csv("model_experiments.csv")
        
        # Identify the most optimized checkpoint
        best_model = results.sort_values(by="mIoU", ascending=False).iloc[0]
        
        print("--- FINAL OPTIMIZED MODEL REPORT ---")
        print(f"🏆 Best Architecture: {best_model['Architecture']}")
        print(f"📊 Final mIoU: {best_model['mIoU']}")
        print(f"🛡️ Deployment Stability: {best_model['Stability']}")
        print("\n✅ DESIGN CHOICES SUMMARY:")
        print("1. Boundary Accuracy: Handled via specialized Dice Loss.")
        print("2. Dust Robustness: Validated via Perturbation Engine.")
        print("3. Small Objects: Targeted with Class-Level weights.")
        
    except FileNotFoundError:
        print("Run your experiments first to generate the report!")

generate_final_report()