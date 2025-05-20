import numpy as np
import sounddevice as sd
import time
import tensorflow as tf
from .model import KWSSystem

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

def main():
    # Initialize KWS system
    kws = KWSSystem(
        sample_rate=16000,
        n_mels=40,
        n_mfcc=13,
        model_type='cnn'  # or 'lstm'
    )
    
    # Load trained model
    try:
        kws.load_model('models/kws_model.h5')
    except:
        print("No trained model found. Please train the model first.")
        return
    
    print("Starting keyword spotting...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            # Capture audio
            audio = capture_audio(duration=1.0)
            
            # Detect keywords
            detections = kws.predict(audio, threshold=0.7)
            
            # Print results
            if detections:
                for detection in detections:
                    print(f"Detected: {detection['keyword']} "
                          f"(confidence: {detection['confidence']:.2f})")
            
            time.sleep(0.1)  # Small delay between captures
            
    except KeyboardInterrupt:
        print("\nStopping keyword spotting")

if __name__ == "__main__":
    main()
