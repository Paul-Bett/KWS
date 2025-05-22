import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue
import threading
import time
from pathlib import Path
from collections import deque

# Define the label mapping
keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_map = {k: i for i, k in enumerate(keywords)}

# Audio parameters
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 0.5  # Reduced duration for faster response
OVERLAP = 0.25  # Reduced overlap

# Feature extraction parameters
WIN_MS = 30
OVERLAP_PERC = 0.25
MAX_FRAMES = 100  # Match training value

# Detection parameters
CONFIDENCE_THRESHOLD = 0.3  # Lowered threshold
SMOOTHING_WINDOW = 3  # Number of consecutive detections required

# Create queues and buffers
audio_queue = queue.Queue()
detection_buffer = deque(maxlen=SMOOTHING_WINDOW)
last_detection = None
last_detection_time = 0
DETECTION_COOLDOWN = 1.0  # Minimum time between detections in seconds

def extract_features(audio_data, sr=SAMPLE_RATE, feature_type='mfcc', win_ms=30, overlap=0.25):
    # Calculate window and hop length
    win_length = int(win_ms * sr / 1000)
    hop_length = int(win_length * (1 - overlap))
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio_data,
        sr=sr,
        n_mfcc=13,
        n_fft=win_length,
        hop_length=hop_length
    )
    
    # Transpose to get (time, features) format
    mfcc = mfcc.T
    
    return mfcc

def predict_keyword(features, model, max_frames=MAX_FRAMES):
    try:
        # Pad or truncate features to match the model's input shape
        if features.shape[0] > max_frames:
            features = features[:max_frames, :]
        else:
            pad_width = max_frames - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        
        # Reshape for model input (batch_size, height, width, channels)
        input_features = np.expand_dims(features, axis=0)  # Add batch dimension
        input_features = np.expand_dims(input_features, axis=-1)  # Add channel dimension
        
        # Make prediction
        predictions = model.predict(input_features, verbose=0)
        predicted_class_index = np.argmax(predictions)
        predicted_keyword = list(label_map.keys())[predicted_class_index]
        
        return predicted_keyword, predictions[0]
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def audio_callback(indata, frames, time, status):
    """Callback function for audio stream"""
    if status:
        print(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())

def update_plot(frame):
    """Update function for animation"""
    global last_detection, last_detection_time
    
    try:
        # Get audio data from queue
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            audio_data = audio_data.flatten()  # Convert to 1D array
            
            # Extract features with exact same parameters as test model
            features = extract_features(audio_data, sr=SAMPLE_RATE, win_ms=30, overlap=0.25)
            
            # Make prediction
            predicted_keyword, predictions = predict_keyword(features, model)
            
            # Update plot
            line.set_ydata(audio_data)
            line.set_xdata(np.arange(len(audio_data)))
            
            # Update prediction text with smoothing
            if predicted_keyword:
                max_prob = np.max(predictions)
                current_time = time.time()
                
                if max_prob > CONFIDENCE_THRESHOLD:
                    detection_buffer.append(predicted_keyword)
                    
                    # Check if we have enough consistent detections
                    if len(detection_buffer) == SMOOTHING_WINDOW and all(x == detection_buffer[0] for x in detection_buffer):
                        # Check cooldown period
                        if current_time - last_detection_time > DETECTION_COOLDOWN:
                            last_detection = detection_buffer[0]
                            last_detection_time = current_time
                            text.set_text(f"Detected: {last_detection} ({max_prob:.2f})")
                        else:
                            text.set_text("Listening...")
                    else:
                        text.set_text("Listening...")
                else:
                    text.set_text("Listening...")
            else:
                text.set_text("Listening...")
            
            # Adjust plot limits
            ax.relim()
            ax.autoscale_view()
            
    except queue.Empty:
        pass
    except Exception as e:
        print(f"Error in update_plot: {e}")
    
    return line, text

def main():
    global model, line, text, ax
    
    # Load the model
    model_path = 'models/keyword_spotting_mfcc_30ms_25ol.h5'
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    line, = ax.plot([], [], lw=2)
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Set up the plot
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, SAMPLE_RATE * DURATION)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Real-time Keyword Spotting')
    ax.grid(True)
    
    # Start audio stream
    print("Starting audio stream...")
    stream = sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * DURATION),
        callback=audio_callback
    )
    
    with stream:
        print("Listening for keywords...")
        print("Press Ctrl+C to stop")
        
        # Start animation
        ani = FuncAnimation(
            fig, update_plot, interval=100,
            blit=True
        )
        plt.show()

if __name__ == "__main__":
    main()
