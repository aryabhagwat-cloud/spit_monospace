import cv2
import numpy as np
import os

# --- PERSON 3: VISUAL AUDIT TOOL ---

def create_overlays(image_dir, mask_dir, output_dir="overlays"):
    """Blends images with masks to check for boundary errors"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define BGR colors for your desert classes
    # Sand: Yellow, Rocks: Red, Bushes: Green
    COLORS = {
        0: [0, 255, 255],  # Sand
        2: [0, 0, 255],    # Rocks
        5: [0, 255, 0]     # Bushes
    }

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg')) and not f.startswith('pred_')]

    for file in image_files:
        # 1. Load the original desert image
        img_path = os.path.join(image_dir, file)
        img = cv2.imread(img_path)
        
        # 2. Look for the corresponding prediction mask
        mask_path = os.path.join(mask_dir, f"pred_{file}")
        
        if not os.path.exists(mask_path):
            print(f"Skipping {file}: No mask found at {mask_path}")
            continue
            
        mask = cv2.imread(mask_path, 0)

        # 3. Create a colored version of the mask
        color_mask = np.zeros_like(img)
        for val, color in COLORS.items():
            color_mask[mask == val] = color

        # 4. Blend the two (0.6 = 60% photo, 0.4 = 40% color mask)
        overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
        
        # 5. Save the visual audit file
        output_path = os.path.join(output_dir, f"overlay_{file}")
        cv2.imwrite(output_path, overlay)
        print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    # Assuming images and masks are both in the 'masks' folder
    create_overlays("masks", "masks")