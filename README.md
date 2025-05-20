# Keyword Spotting (KWS) System

A deep learning-based keyword spotting system using TensorFlow, designed to recognize specific speech commands in real-time. This implementation uses the TensorFlow Speech Commands Dataset and supports both CNN and LSTM architectures.

## Project Structure

```
keyword-spotting/
├── data/                     # Data storage (not committed)
│   ├── train/               # Training audio files
│   ├── val/                 # Validation audio files
│   └── test/                # Test audio files
├── models/                   # Saved model files
│   └── kws_model.h5         # Trained model
├── notebooks/                # Exploratory analysis
│   ├── data_exploration.ipynb    # Audio data analysis
│   └── parameter_tuning.ipynb    # System parameter optimization
├── src/                      # Source code
│   ├── data_loader.py        # Dataset loading utilities
│   ├── feature_extraction.py # Audio preprocessing and feature extraction
│   ├── model.py             # KWS system implementation
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation script
│   ├── live_demo.py         # Real-time demonstration
│   └── __init__.py          # Package initialization
├── tests/                    # Unit tests
├── README.md                 # Project documentation
└── requirements.txt          # Dependencies
```

## Features

- **Deep Learning Models**
  - CNN architecture with multiple convolutional layers
  - LSTM architecture with attention mechanism
  - Support for both MFCC and mel spectrogram features

- **Audio Preprocessing**
  - Bandpass filtering (300Hz - 3000Hz) for speech enhancement
  - Audio normalization for consistent signal levels
  - Voice Activity Detection (VAD) for real-time processing

- **Feature Extraction**
  - Mel spectrogram computation with configurable parameters
  - MFCC extraction with customizable coefficients
  - Configurable window size and overlap

- **Real-time Processing**
  - Live audio capture from microphone
  - Low-latency inference
  - Configurable detection threshold
  - Support for multiple keywords

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Required packages:
- numpy>=1.19.2
- librosa>=0.8.0
- scipy>=1.6.0
- scikit-learn>=0.24.0
- soundfile>=0.10.3
- matplotlib>=3.3.0
- tensorflow>=2.8.0
- sounddevice>=0.4.4
- pandas>=1.3.0
- tqdm>=4.62.0

## Dataset

This implementation uses the TensorFlow Speech Commands Dataset, which contains:
- One-second .wav audio files of spoken English words
- 10 keywords: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'
- Background noise samples for data augmentation

Download the dataset:
```bash
# Using Kaggle API
kaggle competitions download -c tensorflow-speech-recognition-challenge
```

## Quick Start

1. **Prepare the Dataset**
   ```bash
   python -m src.data_loader prepare_dataset
   ```

2. **Train the Model**
   ```bash
   python -m src.train
   ```

3. **Run Live Demo**
   ```bash
   python -m src.live_demo
   ```

## Model Architecture

### CNN Model
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

### LSTM Model
```python
inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
x = tf.keras.layers.Attention()([x, x])
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
```

## Training

The model training process includes:
1. Data augmentation (time shifting, noise addition)
2. Feature extraction (MFCC or mel spectrogram)
3. Model training with early stopping
4. Learning rate reduction on plateau
5. Model evaluation on validation set

## Evaluation

The system is evaluated using:
- Accuracy
- Confusion matrix
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)

## Real-time Processing

The live demonstration includes:
1. Voice Activity Detection
2. Real-time feature extraction
3. Model inference
4. Confidence thresholding
5. Keyword detection output

## Best Practices

1. **Data Preparation**
   - Use high-quality audio recordings
   - Apply data augmentation
   - Balance class distribution
   - Include background noise

2. **Model Selection**
   - CNN for faster inference
   - LSTM for better accuracy
   - Consider model size constraints
   - Balance accuracy vs. latency

3. **System Tuning**
   - Adjust detection threshold
   - Optimize feature parameters
   - Fine-tune model architecture
   - Monitor system performance

## Troubleshooting

1. **Training Issues**
   - Check data quality
   - Verify feature extraction
   - Monitor learning rate
   - Adjust batch size

2. **Inference Problems**
   - Verify model loading
   - Check audio input
   - Adjust VAD threshold
   - Monitor system resources

3. **Performance Issues**
   - Optimize feature extraction
   - Reduce model complexity
   - Use batch processing
   - Profile system bottlenecks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
