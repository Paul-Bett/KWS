import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

def extract_features(audio_data, feature_type='mfcc', win_ms=30, overlap_perc=0.25, sr=16000, n_mels=40):
    win_len = int(sr * win_ms / 1000)
    hop_len = int(win_len * (1 - overlap_perc))
    if feature_type == 'mfcc':
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_fft=win_len, hop_length=hop_len, n_mfcc=13)
        # Pad or truncate to match expected shape
        if mfcc.shape[1] < 98:
            pad_width = 98 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :98]
        return mfcc
    else:
        raise ValueError("Unknown feature type")

def main():
    # Initialize scaler
    scaler = StandardScaler()
    
    # Get all training files
    data_dir = Path('data/train')
    all_features = []
    
    print("Loading training data...")
    for label_dir in data_dir.iterdir():
        if label_dir.is_dir():
            print(f"Processing {label_dir.name}...")
            for audio_file in label_dir.glob('*.wav'):
                try:
                    # Load audio
                    audio_data, sr = librosa.load(str(audio_file), sr=16000)
                    
                    # Extract features
                    features = extract_features(audio_data)
                    
                    # Add to list
                    all_features.append(features.T)  # Transpose for fitting
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
    
    # Stack all features
    print("Fitting scaler...")
    all_features = np.vstack(all_features)
    
    # Fit scaler
    scaler.fit(all_features)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Scaler saved to models/scaler.joblib")

if __name__ == "__main__":
    main() 