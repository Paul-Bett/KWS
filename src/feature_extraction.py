import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from typing import Tuple, Literal

class FeatureExtractor:
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 40,
                 n_mfcc: int = 13,
                 hop_length: int = 160,
                 win_length: int = 400):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.win_length = win_length
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio signal with bandpass filter and normalization.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Preprocessed audio signal
        """
        # Bandpass filter (300Hz - 3000Hz)
        nyquist = self.sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        # Normalize
        normalized_audio = filtered_audio / np.max(np.abs(filtered_audio))
        return normalized_audio
    
    def extract_features(self, 
                        audio: np.ndarray, 
                        feature_type: Literal['mfcc', 'mel'] = 'mfcc') -> np.ndarray:
        """
        Extract audio features (MFCC or Mel spectrogram).
        
        Args:
            audio: Input audio signal
            feature_type: Type of features to extract ('mfcc' or 'mel')
            
        Returns:
            Extracted features
        """
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)
        
        if feature_type == 'mfcc':
            # Extract MFCCs
            features = librosa.feature.mfcc(
                y=processed_audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.win_length,
                hop_length=self.hop_length
            )
        else:
            # Extract mel spectrogram
            features = librosa.feature.melspectrogram(
                y=processed_audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.win_length,
                hop_length=self.hop_length
            )
            # Convert to log scale
            features = librosa.power_to_db(features, ref=np.max)
        
        return features
    
    def detect_voice_activity(self, 
                            audio: np.ndarray, 
                            threshold: float = 0.01) -> bool:
        """
        Basic energy-based voice activity detection.
        
        Args:
            audio: Input audio signal
            threshold: Energy threshold for speech detection
            
        Returns:
            True if speech is detected, False otherwise
        """
        # Calculate short-time energy
        frame_size = int(0.025 * self.sample_rate)  # 25ms frames
        hop_size = int(0.010 * self.sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy.append(np.sum(frame**2) / frame_size)
        
        # Apply threshold
        speech_frames = np.array(energy) > threshold
        
        # Return True if enough frames contain speech
        return np.mean(speech_frames) > 0.3
