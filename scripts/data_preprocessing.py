import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse
import time  
from tqdm import tqdm

def run(args):
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')

    print("----------- Starting Data Preprocessing -----------")
    start_time = time.time()
    
    # Load dataset
    data = []
    labels = []
    classes = 43
    
    # Count total images first for progress tracking
    print("Counting total images...")
    total_images = 0
    for label in range(classes):
        label_dir = os.path.join(train_dir, str(label))
        if os.path.exists(label_dir):
            total_images += len(os.listdir(label_dir))
    
    print(f"Total images to process: {total_images}")
    
    # Load training images with progress bar
    print("Loading and preprocessing images...")
    processed_images = 0
    
    for label in range(classes):
        label_dir = os.path.join(train_dir, str(label))
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} does not exist, skipping...")
            continue
            
        print(f"Processing class {label}...")
        class_images = os.listdir(label_dir)
        
        for img_file in tqdm(class_images, desc=f"Class {label}", leave=False):
            try:
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path, -1)
                
                if img is None:
                    print(f"Warning: Could not read image {img_path}, skipping...")
                    continue
                    
                img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
                data.append(img)
                labels.append(label)
                processed_images += 1
                
                # Print progress every 1000 images
                if processed_images % 1000 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Processed {processed_images}/{total_images} images ({processed_images/total_images*100:.1f}%) - Elapsed: {elapsed_time:.1f}s")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    print(f"Successfully processed {processed_images} images")
    
    if len(data) == 0:
        print("Error: No images were loaded. Please check your data directory structure.")
        return
    
    print("Converting to numpy arrays...")
    data = np.array(data)
    labels = np.array(labels)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Splitting training and testing dataset
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # One hot encoding for labels
    print("One-hot encoding labels...")
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    total_time = time.time() - start_time
    print(f"----------- Data Preprocessing Complete -----------")
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per image: {total_time/processed_images:.4f} seconds") 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing for Traffic Sign Recognition")
    parser.add_argument('--data_dir', type=str, default='data', help='Directory of the data')
    args = parser.parse_args()
    run(args)