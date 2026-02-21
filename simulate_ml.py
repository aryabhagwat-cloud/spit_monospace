import cv2
import numpy as np
import os

# Paths
IMAGE_DIR = "masks" # Use the folder where your photos are
OUTPUT_DIR = "masks"

def simulate_ml_prediction():
    print("🤖 Simulating ML Model Predictions for Person 3 Analytics...")
    
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in files:
        path = os.path.join(IMAGE_DIR, filename)
        img = cv2.imread(path)
        if img is None: continue
        
        # Create a blank mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # SIMULATION LOGIC:
        # In a real demo, this is where Member 1's ML model would output data.
        # We will simulate detecting 'Rocks' (ID 2) in the distance.
        mask[int(h*0.2):int(h*0.5), int(w*0.3):int(w*0.7)] = 2 
        
        # Save as a mask file that batch_analyzer.py can read
        mask_filename = f"pred_{filename.split('.')[0]}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, mask_filename), mask)
        print(f"✅ Generated Prediction Mask: {mask_filename}")

if __name__ == "__main__":
    simulate_ml_prediction()