import numpy as np
import librosa
from model import KWSSystem

def extract_features(audio_data, feature_type='mfcc', win_ms=30, overlap_perc=0.25, sr=16000, n_mels=40):
    win_len = int(sr * win_ms / 1000)
    hop_len = int(win_len * (1 - overlap_perc))
    if feature_type == 'mfcc':
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_fft=win_len, hop_length=hop_len, n_mfcc=13)
        return mfcc
    elif feature_type == 'mel':
        mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_fft=win_len, hop_length=hop_len, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    else:
        raise ValueError("Unknown feature type")

def main():
    # Initialize KWS system
    kws = KWSSystem(sample_rate=16000, n_mels=40, n_mfcc=13, model_type='cnn')
    # Load trained model
    kws.load_model('models/keyword_spotting_mfcc_30ms_25ol.h5')
    # Load a test audio file
    audio_file = 'data/test/yes/3d86b69a_nohash_0.wav'
    audio_data, sr = librosa.load(audio_file, sr=16000)
    print(f"Audio data shape: {audio_data.shape}")
    # Extract features
    features = extract_features(audio_data, feature_type='mfcc')
    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted features (first 5 values): {features[:, :5]}")
    # Get prediction
    detections = kws.predict(features, threshold=0.3)
    if detections:
        keyword = detections[0]['keyword']
        confidence = detections[0]['confidence']
        print(f"Detected keyword '{keyword}' with confidence {confidence:.2f}")
    else:
        print("No keyword detected.")

if __name__ == "__main__":
    main() 