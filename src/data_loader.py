import os
import librosa
import numpy as np
from typing import Tuple, List, Dict
import tensorflow as tf
from pathlib import Path
import requests
from tqdm import tqdm
import tarfile
import shutil

class AudioDataLoader:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Convert to Path object and resolve to absolute path
        file_path = Path(file_path).resolve()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        audio, sr = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        return audio, sr
    
    def load_keyword_template(self, file_path: str) -> np.ndarray:
        """
        Load and process keyword template audio.
        
        Args:
            file_path: Path to the keyword audio file
            
        Returns:
            Processed keyword template
        """
        audio, _ = self.load_audio(file_path)
        return audio
    
    def load_test_audio(self, file_path: str) -> np.ndarray:
        """
        Load and process test audio file.
        
        Args:
            file_path: Path to the test audio file
            
        Returns:
            Processed test audio
        """
        audio, _ = self.load_audio(file_path)
        return audio

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_dataset(self):
        """Download the Speech Commands dataset using TensorFlow."""
        print("Checking for existing dataset...")
        
        # Define the dataset URL and local paths
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
        tar_path = self.data_dir / "speech_commands_v0.02.tar.gz"
        extract_path = self.data_dir
        
        # Check if dataset is already extracted
        required_dirs = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
        dataset_exists = all((extract_path / dir_name).exists() for dir_name in required_dirs)
        
        if dataset_exists:
            print("Dataset already exists and extracted. Skipping download...")
        else:
            print("Downloading Speech Commands dataset...")
            
            # Download the file with progress bar if it doesn't exist
            if not tar_path.exists():
                print(f"Downloading from {dataset_url}")
                response = requests.get(dataset_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(tar_path, 'wb') as f, tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        pbar.update(size)
            
            # Extract the dataset if not already extracted
            if not dataset_exists:
                print("Extracting dataset...")
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=extract_path)
        
        # Get the class names
        label_names = np.array(['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'])
        print("Available commands:", label_names)
        
        # Create dataset from the downloaded files
        train_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=str(extract_path),
            batch_size=64,
            validation_split=0.2,
            seed=0,
            output_sequence_length=16000,
            subset='training'
        )
        
        val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=str(extract_path),
            batch_size=64,
            validation_split=0.2,
            seed=0,
            output_sequence_length=16000,
            subset='validation'
        )
        
        return train_ds, val_ds, label_names
    
    def load_audio_file(self, file_path):
        """Load and preprocess a single audio file."""
        # Read the audio file
        audio = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = tf.reduce_mean(audio, axis=1)
        
        return audio
    
    def get_dataset_info(self):
        """Get information about the downloaded dataset."""
        try:
            train_ds, val_ds, label_names = self.download_dataset()
            
            # Get the actual class names from the dataset
            actual_labels = train_ds.class_names
            
            return {
                'total_classes': len(actual_labels),
                'classes': actual_labels,
                'training_samples': len(list(train_ds)),
                'validation_samples': len(list(val_ds))
            }
        except Exception as e:
            return f"Error getting dataset info: {str(e)}"
