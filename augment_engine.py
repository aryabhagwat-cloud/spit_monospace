import albumentations as A
import cv2
import os
import numpy as np

# --- PERSON 3: RESEARCH AUGMENTATION PIPELINE ---

# 1. Define the transformations
# We include Random Crops, Flips, and Color Jitter as per Responsibility 2
transform = A.Compose([
    A.HorizontalFlip(p=0.5), # 50% chance to flip
    A.RandomBrightnessContrast(p=0.2), # Simulate different sun positions
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5), # Hue shifts
    A.Blur(blur_limit=3, p=0.1), # Noise/Dust injection
])

def run_augmentations(image_path, mask_path, output_dir="aug_samples", num_samples=5):
    """Generates synthetic training variations to help the model generalize"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    for i in range(num_samples):
        # Apply transformation to BOTH image and mask
        augmented = transform(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        # Save the new pair
        cv2.imwrite(os.path.join(output_dir, f"aug_img_{i}.png"), aug_img)
        cv2.imwrite(os.path.join(output_dir, f"aug_mask_{i}.png"), aug_mask)

# Run on your first image as a test
run_augmentations("masks/image1.png", "masks/pred_image1.png")