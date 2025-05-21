import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from .model import KWSSystem
from .feature_extraction import FeatureExtractor
from .data_loader import AudioDataLoader

def create_dataset(data_dir: str, 
                  feature_extractor: FeatureExtractor,
                  batch_size: int = 32,
                  feature_type: str = 'mfcc') -> tf.data.Dataset:
    """
    Create TensorFlow dataset from audio files.
    
    Args:
        data_dir: Directory containing audio files
        feature_extractor: Feature extractor instance
        batch_size: Batch size for training
        feature_type: Type of features to extract ('mfcc' or 'mel')
        
    Returns:
        TensorFlow dataset
    """
    data_loader = AudioDataLoader()
    audio_files = []
    labels = []
    
    # Load audio files and labels
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_files.append(os.path.join(label_dir, audio_file))
                    labels.append(label)
    
    # Create dataset
    def load_and_extract_features(audio_path, label):
        audio = data_loader.load_audio(audio_path)
        features = feature_extractor.extract_features(audio, feature_type)
        return features, label
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((audio_files, labels))
    dataset = dataset.map(
        lambda x, y: tf.py_function(
            load_and_extract_features,
            [x, y],
            [tf.float32, tf.string]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def train_model(data_dir: str,
                model_type: str = 'cnn',
                feature_type: str = 'mfcc',
                batch_size: int = 32,
                epochs: int = 50,
                learning_rate: float = 0.001):
    """
    Train the KWS model.
    
    Args:
        data_dir: Directory containing training data
        model_type: Type of model to use ('cnn' or 'lstm')
        feature_type: Type of features to extract ('mfcc' or 'mel')
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Initial learning rate
    """
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Create datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = create_dataset(
        train_dir,
        feature_extractor,
        batch_size,
        feature_type
    )
    val_dataset = create_dataset(
        val_dir,
        feature_extractor,
        batch_size,
        feature_type
    )
    
    # Initialize KWS system
    kws = KWSSystem(
        model_type=model_type,
        n_mfcc=13 if feature_type == 'mfcc' else None
    )
    
    # Train model
    history = kws.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    kws.save_model('models/kws_model.h5')
    
    return history

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train KWS model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--model_type', type=str, default='cnn',
                      choices=['cnn', 'lstm'],
                      help='Type of model to use')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                      choices=['mfcc', 'mel'],
                      help='Type of features to extract')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    
    args = parser.parse_args()
    
    # Train model
    history = train_model(
        args.data_dir,
        model_type=args.model_type,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print("Training completed. Model saved to models/kws_model.h5")

if __name__ == "__main__":
    main()
