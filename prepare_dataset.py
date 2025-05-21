import os
import shutil
import tarfile
from pathlib import Path
import numpy as np

def prepare_dataset(data_dir='data'):
    """
    Prepare the dataset by organizing it into train, val, and test directories.
    
    Args:
        data_dir: Directory containing the raw dataset
    """
    data_dir = Path(data_dir)
    
    # Create train, val, and test directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Create directories if they don't exist
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Read validation and test lists
    with open(data_dir / 'validation_list.txt', 'r') as f:
        validation_files = [line.strip() for line in f.readlines()]
    print('First 5 validation_list.txt entries:', validation_files[:5])
    
    with open(data_dir / 'testing_list.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    print('First 5 testing_list.txt entries:', test_files[:5])
    
    # Only process these 10 classes
    required_classes = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
    # Process each class directory
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and class_dir.name in required_classes:
            print(f"Processing class: {class_dir.name}")
            # Create class directories in train, val, and test
            for split_dir in [train_dir, val_dir, test_dir]:
                (split_dir / class_dir.name).mkdir(exist_ok=True)
            # Print first 5 rel_path values for this class
            rel_paths = []
            for i, audio_file in enumerate(class_dir.glob('*.wav')):
                rel_path = audio_file.relative_to(data_dir)
                if i < 5:
                    rel_paths.append(str(rel_path))
            print(f"First 5 rel_path values for {class_dir.name}: {rel_paths}")
            # Move files to appropriate directories
            for audio_file in class_dir.glob('*.wav'):
                rel_path = audio_file.relative_to(data_dir)
                rel_path_str = rel_path.as_posix()
                target_dir = None
                if rel_path_str in validation_files:
                    target_dir = val_dir / class_dir.name
                elif rel_path_str in test_files:
                    target_dir = test_dir / class_dir.name
                else:
                    target_dir = train_dir / class_dir.name
                # Copy file if it exists in source and doesn't exist in target directory
                target_file = target_dir / audio_file.name
                if audio_file.exists():
                    if not target_file.exists():
                        try:
                            shutil.copy2(audio_file, target_file)
                        except Exception as e:
                            print(f"Error copying {audio_file}: {e}")
                else:
                    print(f"Warning: Source file does not exist: {audio_file}")
    
    print("\nDataset preparation completed!")
    print(f"Training samples: {sum(len(list((train_dir / d).glob('*.wav'))) for d in train_dir.iterdir() if d.is_dir())}")
    print(f"Validation samples: {sum(len(list((val_dir / d).glob('*.wav'))) for d in val_dir.iterdir() if d.is_dir())}")
    print(f"Test samples: {sum(len(list((test_dir / d).glob('*.wav'))) for d in test_dir.iterdir() if d.is_dir())}")

if __name__ == "__main__":
    prepare_dataset() 