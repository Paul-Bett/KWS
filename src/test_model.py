import numpy as np
import tensorflow as tf
import librosa
import os
from pathlib import Path
import random

# Define the label mapping
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_map = {k: i for i, k in enumerate(keywords)}

# Function to extract features from audio file
def extract_features(audio_file, feature_type='mfcc', win_ms=30, overlap=0.25, max_frames=100):
    # Load audio file
    audio_data, sr = librosa.load(audio_file, sr=16000)
    
    # Calculate window and hop length
    win_length = int(win_ms * sr / 1000)
    hop_length = int(win_length * (1 - overlap))
    
    # Extract MFCC features (shape: n_mfcc, time_frames)
    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sr,
        n_mfcc=13,
        n_fft=win_length,
        hop_length=hop_length
    )
    # Pad or truncate the second dimension (time frames) to max_frames
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    return mfcc

# Function to make a prediction on a single audio file
def predict_keyword(audio_file_path, model, feature_type='mfcc', win_ms=30, overlap=0.25, max_frames=100):
    try:
        # Extract features from the audio file
        features = extract_features(audio_file_path, feature_type, win_ms, overlap, max_frames)
        print(f"Extracted features shape: {features.shape}")

        # Reshape for model input (batch_size, height, width, channels)
        input_features = np.expand_dims(features, axis=0) # Add batch dimension
        input_features = np.expand_dims(input_features, axis=-1) # Add channel dimension
        print(f"Final input shape: {input_features.shape}")

        # Make prediction
        predictions = model.predict(input_features)
        predicted_class_index = np.argmax(predictions)
        predicted_keyword = list(label_map.keys())[predicted_class_index]

        return predicted_keyword, predictions[0]

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
        return None, None

def main():
    # Load the model
    model_path = 'models/keyword_spotting_mfcc_30ms_25ol.h5'
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Select a random audio file from the test set for inference
    test_dir = 'data/test'
    test_files = []
    for keyword in keywords:
        keyword_dir = os.path.join(test_dir, keyword)
        if os.path.exists(keyword_dir):
            for file in os.listdir(keyword_dir):
                if file.endswith('.wav'):
                    test_files.append((os.path.join(keyword_dir, file), keyword))

    if test_files:
        random_test_file, true_label = random.choice(test_files)
        print(f"\n🚀 Inferencing on file: {random_test_file}")
        print(f"True label: {true_label}")

        # Make a prediction
        predicted_keyword, predictions = predict_keyword(random_test_file, model)

        if predicted_keyword:
            print(f"Predicted keyword: {predicted_keyword}")
            print(f"Prediction probabilities: {predictions}")
            # Optional: map probabilities to keywords
            prob_dict = {kw: prob for kw, prob in zip(keywords, predictions)}
            print("Probabilities per keyword:")
            for kw, prob in prob_dict.items():
                print(f"  {kw}: {prob:.4f}")
        else:
            print("Failed to make prediction.")
    else:
        print("Test set is empty. Cannot perform inference.")

if __name__ == "__main__":
    main() 