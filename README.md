# Keyword Spotting (KWS) System

This is a simple Keyword Spotting system that uses mel spectrogram features and template matching to detect keywords in audio signals. The system is designed to detect specific spoken words or phrases in continuous audio streams.

## Features

- Audio preprocessing with bandpass filtering (300Hz - 3000Hz)
- Mel spectrogram feature extraction
- Template matching using cosine similarity
- Visualization of detections with waveform and spectrogram
- Configurable parameters for feature extraction

## Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Implementation Steps

### 1. Audio Preprocessing
The preprocessing stage involves two main steps:

1. **Bandpass Filtering (300Hz - 3000Hz)**
   - Removes frequencies outside the speech range
   - Reduces background noise
   - Improves signal-to-noise ratio
   - Uses a 4th-order Butterworth filter
   - Implementation: `preprocess_audio()` method

2. **Audio Normalization**
   - Scales audio amplitude to [-1, 1] range
   - Ensures consistent signal levels
   - Helps in feature extraction stability

### 2. Feature Extraction
The system uses Mel spectrograms as features:

1. **Mel Spectrogram Computation**
   - Converts audio to frequency domain using STFT
   - Applies mel-scale filterbank (40 bands)
   - Parameters:
     - Window length: 400 samples
     - Hop length: 160 samples
     - Sample rate: 16000 Hz
   - Implementation: `extract_features()` method

2. **Log-scale Conversion**
   - Converts power spectrogram to decibel scale
   - Improves feature representation
   - Better matches human auditory perception

### 3. Keyword Detection
The detection process uses template matching:

1. **Template Creation**
   - Uses a clean recording of the keyword
   - Extracts mel spectrogram features
   - Stores as reference template
   - Implementation: `set_keyword_template()` method

2. **Template Matching**
   - Uses sliding window approach
   - Computes cosine similarity between template and audio segments
   - Parameters:
     - Window size: matches template length
     - Step size: 1 frame
   - Implementation: `detect_keyword()` method

3. **Peak Detection and Grouping**
   - Identifies similarity scores above threshold
   - Groups nearby detections
   - Prevents multiple detections of same occurrence
   - Parameters:
     - Threshold: 0.7 (adjustable)
     - Grouping window: template length

### 4. Visualization
The system provides visual feedback:

1. **Waveform Display**
   - Shows audio waveform
   - Marks detection points
   - Helps verify detection accuracy

2. **Spectrogram Display**
   - Shows mel spectrogram
   - Visualizes frequency content
   - Aids in understanding detection process

## Usage

1. Prepare your audio files:
   - `keyword.wav`: A clean recording of the keyword you want to detect
   - `test_audio.wav`: The audio file in which you want to detect the keyword

2. Run the system:

```bash
python kws_system.py
```

## Technical Details

### Parameters
You can adjust the following parameters in the `KWSSystem` class:

- `sample_rate`: Audio sampling rate (default: 16000 Hz)
  - Standard rate for speech processing
  - Balances quality and computational cost

- `n_mels`: Number of mel bands (default: 40)
  - Represents frequency resolution
  - Higher values capture more detail but increase computation

- `hop_length`: Number of samples between frames (default: 160)
  - Controls temporal resolution
  - Smaller values give better time precision

- `win_length`: Window length for STFT (default: 400)
  - Affects frequency resolution
  - Longer windows give better frequency resolution

### Example Code

```python
from kws_system import KWSSystem

# Initialize the system
kws = KWSSystem()

# Load and set keyword template
keyword_audio, sr = librosa.load('keyword.wav', sr=16000)
kws.set_keyword_template(keyword_audio)

# Load test audio
test_audio, sr = librosa.load('test_audio.wav', sr=16000)

# Detect keywords
detections = kws.detect_keyword(test_audio, threshold=0.7)

# Visualize results
kws.visualize_detection(test_audio, detections)
```

## Performance Considerations

1. **Accuracy Factors**
   - Quality of keyword template
   - Background noise levels
   - Speaker variations
   - Speaking rate variations

2. **Computational Efficiency**
   - Template matching is computationally intensive
   - Processing time scales with audio length
   - Consider batch processing for long audio

3. **Memory Usage**
   - Mel spectrograms require significant memory
   - Consider processing in chunks for long audio

## Best Practices

1. **Template Recording**
   - Use clear, isolated keyword recordings
   - Record in similar conditions to test audio
   - Consider multiple templates for robustness

2. **Threshold Selection**
   - Start with default (0.7)
   - Adjust based on:
     - False positive rate
     - False negative rate
     - Application requirements

3. **Audio Quality**
   - Use high-quality recordings
   - Minimize background noise
   - Maintain consistent recording conditions

## Notes

- The system works best with clear, isolated keyword recordings
- Adjust the threshold parameter based on your needs (default: 0.7)
- The bandpass filter is optimized for speech signals
- The visualization helps in understanding the detection process
- Consider using multiple templates for better robustness
- Regular evaluation and threshold adjustment may be needed
