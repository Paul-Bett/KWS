import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from .model import KWSSystem
from .feature_extraction import FeatureExtractor
from .data_loader import AudioDataLoader

def evaluate_model(model_path: str,
                  test_dir: str,
                  feature_type: str = 'mfcc',
                  batch_size: int = 32) -> dict:
    """
    Evaluate the trained KWS model.
    
    Args:
        model_path: Path to the trained model
        test_dir: Directory containing test data
        feature_type: Type of features to extract ('mfcc' or 'mel')
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model
    kws = KWSSystem()
    kws.load_model(model_path)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor()
    
    # Load test data
    data_loader = AudioDataLoader()
    audio_files = []
    true_labels = []
    
    for label in os.listdir(test_dir):
        label_dir = os.path.join(test_dir, label)
        if os.path.isdir(label_dir):
            for audio_file in os.listdir(label_dir):
                if audio_file.endswith('.wav'):
                    audio_files.append(os.path.join(label_dir, audio_file))
                    true_labels.append(label)
    
    # Get predictions
    predictions = []
    confidences = []
    
    for audio_file in audio_files:
        audio = data_loader.load_audio(audio_file)
        detections = kws.predict(audio)
        
        if detections:
            predictions.append(detections[0]['keyword'])
            confidences.append(detections[0]['confidence'])
        else:
            predictions.append('unknown')
            confidences.append(0.0)
    
    # Calculate metrics
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Calculate FAR and FRR
    far = calculate_far(cm, kws.labels)
    frr = calculate_frr(cm, kws.labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=kws.labels,
                yticklabels=kws.labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'far': far,
        'frr': frr,
        'accuracy': report['accuracy']
    }

def calculate_far(cm: np.ndarray, labels: list) -> float:
    """
    Calculate False Acceptance Rate.
    
    Args:
        cm: Confusion matrix
        labels: List of class labels
        
    Returns:
        False Acceptance Rate
    """
    # Get index of 'unknown' class
    unknown_idx = labels.index('unknown')
    
    # Calculate FAR
    false_accepts = np.sum(cm[unknown_idx, :]) - cm[unknown_idx, unknown_idx]
    total_unknown = np.sum(cm[unknown_idx, :])
    
    return false_accepts / total_unknown if total_unknown > 0 else 0.0

def calculate_frr(cm: np.ndarray, labels: list) -> float:
    """
    Calculate False Rejection Rate.
    
    Args:
        cm: Confusion matrix
        labels: List of class labels
        
    Returns:
        False Rejection Rate
    """
    # Get index of 'unknown' class
    unknown_idx = labels.index('unknown')
    
    # Calculate FRR
    false_rejects = np.sum(cm[:, unknown_idx]) - cm[unknown_idx, unknown_idx]
    total_keywords = np.sum(cm) - np.sum(cm[unknown_idx, :])
    
    return false_rejects / total_keywords if total_keywords > 0 else 0.0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate KWS model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--feature_type', type=str, default='mfcc',
                      choices=['mfcc', 'mel'],
                      help='Type of features to extract')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        args.model_path,
        args.test_dir,
        feature_type=args.feature_type,
        batch_size=args.batch_size
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"False Acceptance Rate: {metrics['far']:.4f}")
    print(f"False Rejection Rate: {metrics['frr']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        metrics['classification_report'],
        target_names=metrics['classification_report'].keys()
    ))
    print("\nConfusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    main()
