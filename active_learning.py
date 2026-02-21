import os
import shutil
import cv2

def active_learning_trigger(image, image_name, is_edge_case, confidence):
    save_path = "data_factory/to_be_labeled"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if is_edge_case:
        # Save the image that confused the AI
        file_path = os.path.join(save_path, f"low_conf_{confidence:.2f}_{image_name}")
        cv2.imwrite(file_path, image)
        print(f"⚠️ Edge Case Detected! Image saved for retraining: {file_path}")