import numpy as np
import librosa
from model import KWSSystem
from sklearn.preprocessing import StandardScaler

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
    
    # Print model summary
    print("\nModel Summary:")
    kws.model.summary()
    
    # Load a test audio file
    audio_file = 'data/train/up/0a7c2a8d_nohash_0.wav'
    audio_data, sr = librosa.load(audio_file, sr=16000)
    print(f"\nAudio data shape: {audio_data.shape}")
    
    # Extract features
    features = extract_features(audio_data, feature_type='mfcc')
    print(f"Extracted features shape: {features.shape}")
    print(f"Extracted features (first 5 values): {features[:, :5]}")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features.T).T
    print(f"Normalized features (first 5 values): {features_normalized[:, :5]}")
    
    # Reshape features for model input
    features_reshaped = np.expand_dims(features_normalized, axis=0)
    features_reshaped = np.expand_dims(features_reshaped, axis=-1)
    print(f"\nReshaped features shape: {features_reshaped.shape}")
    
    # Get raw predictions
    raw_predictions = kws.model.predict(features_reshaped)
    print(f"\nRaw predictions shape: {raw_predictions.shape}")
    print(f"Raw predictions: {raw_predictions}")

    # Print max probability and its class
    max_idx = np.argmax(raw_predictions[0])
    max_prob = raw_predictions[0][max_idx]
    class_names = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    print(f"\nMax probability: {max_prob:.4f} for class: '{class_names[max_idx]}' (index {max_idx})")
    
    # Print all classes above threshold
    print(f"\nClasses above threshold {0.1}:")
    for idx, prob in enumerate(raw_predictions[0]):
        if prob > 0.1:
            print(f"  Class '{class_names[idx]}' (index {idx}): {prob:.4f}")

    # Get prediction with threshold
    detections = kws.predict(features_normalized, threshold=0.3)
    if detections:
        keyword = detections[0]['keyword']
        confidence = detections[0]['confidence']
        print(f"\nDetected keyword '{keyword}' with confidence {confidence:.2f}")
    else:
        print("\nNo keyword detected.")

if __name__ == "__main__":
    main() 