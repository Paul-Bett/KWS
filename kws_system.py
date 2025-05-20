import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.metrics.pairwise import cosine_similarity

class KWSSystem:
    def __init__(self, sample_rate=16000, n_mels=40, hop_length=160, win_length=400):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.keyword_template = None
        
    def preprocess_audio(self, audio):
        """Preprocess audio signal with bandpass filter and normalization."""
        # Bandpass filter (300Hz - 3000Hz)
        nyquist = self.sample_rate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        # Normalize
        normalized_audio = filtered_audio / np.max(np.abs(filtered_audio))
        return normalized_audio
    
    def extract_features(self, audio):
        """Extract mel spectrogram features from audio."""
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=processed_audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def set_keyword_template(self, audio):
        """Set the keyword template from reference audio."""
        self.keyword_template = self.extract_features(audio)
    
    def detect_keyword(self, audio, threshold=0.7):
        """Detect keyword in audio using template matching."""
        if self.keyword_template is None:
            raise ValueError("Keyword template not set. Call set_keyword_template first.")
        
        # Extract features from input audio
        features = self.extract_features(audio)
        
        # Compute similarity using sliding window
        template_len = self.keyword_template.shape[1]
        similarities = []
        
        for i in range(features.shape[1] - template_len + 1):
            window = features[:, i:i+template_len]
            similarity = cosine_similarity(
                self.keyword_template.T,
                window.T
            )[0][0]
            similarities.append(similarity)
        
        # Find peaks above threshold
        similarities = np.array(similarities)
        peaks = np.where(similarities > threshold)[0]
        
        # Group nearby peaks
        if len(peaks) > 0:
            groups = []
            current_group = [peaks[0]]
            
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] <= template_len:
                    current_group.append(peaks[i])
                else:
                    groups.append(current_group)
                    current_group = [peaks[i]]
            groups.append(current_group)
            
            # Get the highest similarity peak from each group
            detections = []
            for group in groups:
                max_idx = group[np.argmax(similarities[group])]
                detections.append({
                    'position': max_idx * self.hop_length / self.sample_rate,
                    'similarity': similarities[max_idx]
                })
            
            return detections
        
        return []
    
    def visualize_detection(self, audio, detections):
        """Visualize the audio waveform and detections."""
        plt.figure(figsize=(15, 5))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(audio, sr=self.sample_rate)
        plt.title('Audio Waveform with Keyword Detections')
        
        # Mark detections
        for detection in detections:
            plt.axvline(x=detection['position'], color='r', linestyle='--')
        
        # Plot mel spectrogram
        plt.subplot(2, 1, 2)
        mel_spec = self.extract_features(audio)
        librosa.display.specshow(
            mel_spec,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    kws = KWSSystem()
    
    # Load reference keyword audio
    keyword_audio, sr = librosa.load('keyword.wav', sr=16000)
    kws.set_keyword_template(keyword_audio)
    
    # Load test audio
    test_audio, sr = librosa.load('test_audio.wav', sr=16000)
    
    # Detect keywords
    detections = kws.detect_keyword(test_audio, threshold=0.7)
    
    # Print detections
    for detection in detections:
        print(f"Keyword detected at {detection['position']:.2f}s with similarity {detection['similarity']:.2f}")
    
    # Visualize results
    kws.visualize_detection(test_audio, detections)

if __name__ == "__main__":
    main() 