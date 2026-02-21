import pandas as pd
import os
import cv2  # <--- Add this line!
from research_tools import PerceptionEvaluator

# Settings
MASK_DIR = "masks"
evaluator = PerceptionEvaluator({2: "Rock", 3: "Log", 4: "Tree", 5: "Bush"})

def run_advanced_benchmark():
    report = []
    files = [f for f in os.listdir(MASK_DIR) if f.startswith('pred_')]
    
    for f in files:
        mask = cv2.imread(os.path.join(MASK_DIR, f), 0)
        
        # 1. Deployment Stability Index (Standard deviation simulation)
        safety = evaluator.get_safety_score(mask)
        complexity = evaluator.get_navigation_complexity(mask)
        
        # 2. Hard-Case Clustering
        category = "Safe"
        if safety < 50: category = "Technical Failure Risk"
        elif complexity > 10: category = "Visual Clutter Failure"

        report.append({
            "Image": f,
            "Safety_Score": safety,
            "Cluster_Density": complexity,
            "Hard_Case_Type": category,
            "Planner_Effort_Index": round(complexity * (100-safety)/100, 2)
        })

    df = pd.DataFrame(report)
    df.to_csv("advanced_benchmark_report.csv", index=False)
    print("✅ Advanced Benchmark Report Generated.")

if __name__ == "__main__":
    run_advanced_benchmark()