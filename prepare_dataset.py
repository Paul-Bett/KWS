import os
import shutil
from pathlib import Path

def prepare_dataset(data_dir='data'):
    # Create train, val, and test directories if they don't exist
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_dir, split), exist_ok=True)
    
    # Read validation and test lists
    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as f:
        validation_files = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(data_dir, 'testing_list.txt'), 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    # Convert lists to sets for faster lookups
    validation_files = set(validation_files)
    test_files = set(test_files)
    
    # Process each class directory
    classes = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    train_count = 0
    val_count = 0
    test_count = 0
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Class directory {class_dir} does not exist")
            continue
            
        # Create class subdirectories in train, val, and test
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(data_dir, split, class_name), exist_ok=True)
        
        # Process each audio file
        for audio_file in os.listdir(class_dir):
            if not audio_file.endswith('.wav'):
                continue
                
            # Get relative path with forward slashes
            rel_path = f"{class_name}/{audio_file}"
            
            # Determine target directory based on file lists
            if rel_path in validation_files:
                target_dir = os.path.join(data_dir, 'val', class_name)
                val_count += 1
            elif rel_path in test_files:
                target_dir = os.path.join(data_dir, 'test', class_name)
                test_count += 1
            else:
                target_dir = os.path.join(data_dir, 'train', class_name)
                train_count += 1
            
            # Copy file to target directory
            source_path = os.path.join(class_dir, audio_file)
            target_path = os.path.join(target_dir, audio_file)
            
            # Print debug info
            print(f"Processing: {rel_path}")
            print(f"Source exists: {os.path.exists(source_path)}")
            print(f"Target exists: {os.path.exists(target_path)}")
            
            # Copy file
            shutil.copy2(source_path, target_path)
            print(f"Copied to: {target_path}")
    
    print(f"\nDataset preparation completed:")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")

if __name__ == '__main__':
    prepare_dataset() 