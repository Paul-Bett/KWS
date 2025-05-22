import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from model import KWSSystem
from feature_extraction import FeatureExtractor
from data_loader import AudioDataLoader

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
    
    # Convert to Path object
    data_dir = Path(data_dir)
    
    # Load audio files and labels
    for label_dir in data_dir.iterdir():
        if label_dir.is_dir():
            for audio_file in label_dir.glob('*.wav'):
                if audio_file.exists():
                    audio_files.append(str(audio_file))
                    labels.append(label_dir.name)
    
    print(f"Found {len(audio_files)} audio files in {data_dir}")
    
    # Create dataset
    def load_and_extract_features(audio_path, label):
        # Convert EagerTensor to Python string
        if hasattr(audio_path, 'numpy'):
            audio_path = audio_path.numpy().decode('utf-8')
        if hasattr(label, 'numpy'):
            label = label.numpy().decode('utf-8')
        audio, _ = data_loader.load_audio(audio_path)  # Unpack the tuple
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

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    
    print(f"Loading training data from {train_dir}")
    train_dataset = create_dataset(
        str(train_dir),
        feature_extractor,
        batch_size,
        feature_type
    )
    
    print(f"Loading validation data from {val_dir}")
    val_dataset = create_dataset(
        str(val_dir),
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
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    kws.save_model('models/kws_model.h5')
    
    # Plot and save training history
    plot_training_history(history, 'models/training_history.png')
    
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
