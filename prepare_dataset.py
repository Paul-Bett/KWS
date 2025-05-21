import os
import shutil
import tarfile
from pathlib import Path
import numpy as np

def prepare_dataset(data_dir='data'):
    """
    Prepare the dataset by extracting and organizing it into train, val, and test directories.
    
    Args:
        data_dir: Directory containing the raw dataset
    """
    data_dir = Path(data_dir)
    
    # Extract the dataset if needed
    tar_path = data_dir / "speech_commands_v0.02.tar.gz"
    if tar_path.exists():
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
    
    # Create train, val, and test directories
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Read validation and test lists
    with open(data_dir / 'validation_list.txt', 'r') as f:
        validation_files = [line.strip() for line in f.readlines()]
    
    with open(data_dir / 'testing_list.txt', 'r') as f:
        test_files = [line.strip() for line in f.readlines()]
    
    # Process each class directory
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir() and not class_dir.name.startswith('_') and class_dir.name not in ['train', 'val', 'test']:
            # Create class directories in train, val, and test
            for split_dir in [train_dir, val_dir, test_dir]:
                (split_dir / class_dir.name).mkdir(exist_ok=True)
            
            # Move files to appropriate directories
            for audio_file in class_dir.glob('*.wav'):
                rel_path = audio_file.relative_to(data_dir)
                if str(rel_path) in validation_files:
                    shutil.copy2(audio_file, val_dir / class_dir.name / audio_file.name)
                elif str(rel_path) in test_files:
                    shutil.copy2(audio_file, test_dir / class_dir.name / audio_file.name)
                else:
                    shutil.copy2(audio_file, train_dir / class_dir.name / audio_file.name)
    
    print("Dataset preparation completed!")
    print(f"Training samples: {sum(len(list((train_dir / d).glob('*.wav'))) for d in train_dir.iterdir() if d.is_dir())}")
    print(f"Validation samples: {sum(len(list((val_dir / d).glob('*.wav'))) for d in val_dir.iterdir() if d.is_dir())}")
    print(f"Test samples: {sum(len(list((test_dir / d).glob('*.wav'))) for d in test_dir.iterdir() if d.is_dir())}")

if __name__ == "__main__":
    prepare_dataset() 