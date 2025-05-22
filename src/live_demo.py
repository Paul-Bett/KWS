import numpy as np
import sounddevice as sd
import time
import tensorflow as tf
from model import KWSSystem
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
import librosa
from sklearn.preprocessing import StandardScaler
import joblib

# Global variables for audio processing
audio_queue = queue.Queue()
is_recording = False

def capture_audio(duration: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Capture audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sampling rate
        
    Returns:
        Recorded audio signal
    """
    print("Recording...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32
    )
    sd.wait()
    print("Done recording")
    return audio.flatten()

def audio_callback(indata, frames, time, status):
    """Callback for continuous audio stream."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

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

def update_plot(frame, line, audio_data, kws, ax, scaler):
    """Update the plot with new audio data and predictions."""
    try:
        # Get audio data from queue
        while not audio_queue.empty():
            audio_data = audio_queue.get().flatten()
        
        # Update audio plot
        x = np.arange(len(audio_data))
        line.set_data(x, audio_data)
        
        # Extract features
        features = extract_features(audio_data, feature_type='mfcc')
        
        # Normalize features using pre-fitted scaler
        features_normalized = scaler.transform(features.T).T
        
        # Reshape for model input
        features_reshaped = np.expand_dims(features_normalized, axis=0)
        features_reshaped = np.expand_dims(features_reshaped, axis=-1)
        
        # Get prediction
        detections = kws.predict(features_reshaped, threshold=0.3)
        
        # Update title with detection
        if detections:
            keyword = detections[0]['keyword']
            confidence = detections[0]['confidence']
            ax.set_title(f"Detected: {keyword} (confidence: {confidence:.2f})", color='green')
            print(f"Detected keyword '{keyword}' with confidence {confidence:.2f}")
        else:
            ax.set_title("Listening...", color='blue')
            print("Listening...")
        return line,
    except Exception as e:
        print(f"Error in update_plot: {e}")
        return line,

def main():
    # Initialize KWS system
    kws = KWSSystem(
        sample_rate=16000,
        n_mels=40,
        n_mfcc=13,
        model_type='cnn'
    )
    
    # Load trained model
    try:
        kws.load_model('models/keyword_spotting_mfcc_30ms_25ol.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load pre-fitted scaler
    try:
        scaler = joblib.load('models/scaler.joblib')
        print("Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return
    
    print("Starting keyword spotting...")
    print("Press Ctrl+C to stop")
    
    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(16000)
    audio_data = np.zeros(16000)
    line, = ax.plot(x, audio_data, lw=2)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 16000)  # 1 second of audio at 16kHz
    ax.set_title("Listening...")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    
    try:
        # Start audio stream
        with sd.InputStream(callback=audio_callback,
                          channels=1,
                          samplerate=16000,
                          blocksize=16000):
            # Create animation
            ani = FuncAnimation(fig, update_plot, fargs=(line, audio_data, kws, ax, scaler),
                              interval=100, blit=True)
            plt.show()
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    main()
