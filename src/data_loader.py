import os
import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple
import shutil
from tqdm import tqdm

class AudioDataLoader:
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio data loader.
        
        Args:
            sample_rate: Target sample rate for audio files
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal as numpy array
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def prepare_dataset(self, 
                       data_dir: str,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1) -> None:
        """
        Prepare dataset by splitting into train/val/test sets.
        
        Args:
            data_dir: Directory containing raw audio files
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
        """
        # Create output directories
        os.makedirs(os.path.join(data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test'), exist_ok=True)
        
        # Process each keyword directory
        for keyword in os.listdir(data_dir):
            keyword_dir = os.path.join(data_dir, keyword)
            if os.path.isdir(keyword_dir) and keyword not in ['train', 'val', 'test']:
                # Get all audio files
                audio_files = [f for f in os.listdir(keyword_dir) 
                             if f.endswith('.wav')]
                
                # Shuffle files
                np.random.shuffle(audio_files)
                
                # Calculate split indices
                n_files = len(audio_files)
                n_train = int(n_files * train_ratio)
                n_val = int(n_files * val_ratio)
                
                # Split files
                train_files = audio_files[:n_train]
                val_files = audio_files[n_train:n_train + n_val]
                test_files = audio_files[n_train + n_val:]
                
                # Create keyword directories
                for split in ['train', 'val', 'test']:
                    os.makedirs(os.path.join(data_dir, split, keyword), 
                              exist_ok=True)
                
                # Copy files to respective directories
                for files, split in [(train_files, 'train'),
                                   (val_files, 'val'),
                                   (test_files, 'test')]:
                    for file in tqdm(files, desc=f'Copying {split} files'):
                        src = os.path.join(keyword_dir, file)
                        dst = os.path.join(data_dir, split, keyword, file)
                        shutil.copy2(src, dst)
                
                # Remove original directory
                shutil.rmtree(keyword_dir)
    
    def download_dataset(self, output_dir: str) -> None:
        """
        Download TensorFlow Speech Commands dataset.
        
        Args:
            output_dir: Directory to save the dataset
        """
        import kaggle
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download dataset
        print("Downloading dataset...")
        kaggle.api.competition_download_files(
            'tensorflow-speech-recognition-challenge',
            path=output_dir
        )
        
        # Extract files
        print("Extracting files...")
        import zipfile
        with zipfile.ZipFile(
            os.path.join(output_dir, 'train.7z'), 'r'
        ) as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Prepare dataset
        print("Preparing dataset...")
        self.prepare_dataset(output_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare KWS dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Directory to save the dataset')
    parser.add_argument('--download', action='store_true',
                      help='Download dataset from Kaggle')
    
    args = parser.parse_args()
    
    data_loader = AudioDataLoader()
    
    if args.download:
        data_loader.download_dataset(args.output_dir)
    else:
        data_loader.prepare_dataset(args.output_dir)
    
    print("Dataset preparation completed.")

if __name__ == "__main__":
    main()
