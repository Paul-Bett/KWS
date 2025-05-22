# Keyword Spotting System

This project implements a real-time keyword spotting system using deep learning. The system can detect 10 different keywords: "yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

## Features

- Real-time audio processing and keyword detection
- Live visualization of audio waveform and predictions
- Support for 10 different keywords
- Configurable detection parameters
- Pre-trained model included

## Requirements

- Python 3.8+
- TensorFlow 2.x
- librosa
- sounddevice
- numpy
- matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd keyword-spotting
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Live Demo

To run the live demo with real-time audio processing:

```bash
python src/live_demo.py
```

The live demo features:
- Real-time audio visualization
- Continuous keyword detection
- Confidence scores for all keywords
- Smoothing to reduce false positives
- Configurable detection parameters

### Test Model

To test the model on a random audio file from the test set:

```bash
python src/test_model.py
```

The test script will:
- Load a random audio file from the test set
- Process the audio and extract features
- Make predictions using the trained model
- Display the predicted keyword and confidence scores

## Model Architecture

The system uses a CNN-based model trained on MFCC features extracted from audio data. The model architecture includes:
- Input shape: (time_steps, features, channels)
- Convolutional layers for feature extraction
- Dense layers for classification
- Softmax output for 10 keyword classes

## Configuration

Key parameters can be adjusted in the source files:

- `SAMPLE_RATE`: Audio sampling rate (default: 16000 Hz)
- `WIN_MS`: Window size for feature extraction (default: 30ms)
- `OVERLAP_PERC`: Overlap percentage between windows (default: 25%)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detection (default: 0.2)
- `SMOOTHING_WINDOW`: Number of consistent detections required (default: 2)
- `DETECTION_COOLDOWN`: Minimum time between detections (default: 0.5s)

## Training

The model was trained on the Speech Commands dataset using the following parameters:
- MFCC features with 13 coefficients
- 30ms window size with 25% overlap
- Data augmentation including time shifting and background noise
- Adam optimizer with categorical crossentropy loss

## License

[Your License Here]

## Acknowledgments

- Speech Commands dataset
- TensorFlow team
- Librosa developers
