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
│   └── keyword_spotting_mfcc_30ms_25ol.h5  # Trained model
├── notebooks/                # Jupyter notebooks
│   ├── KWS_Training_mfcc_30ms_25ol.ipynb           # Main training notebook
│   ├── KWS_Training_Evaluation_Chaging_Parameters.ipynb  # Parameter evaluation
│   ├── kws_training.ipynb   # Initial training experiments
│   ├── data_exploration.ipynb    # Audio data analysis
│   └── parameter_tuning.ipynb    # System parameter optimization
├── src/                      # Source code
│   ├── data_loader.py        # Dataset loading utilities
│   ├── feature_extraction.py # Audio preprocessing and feature extraction
│   ├── model.py             # KWS system implementation
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Model evaluation script
│   ├── live_demo.py         # Real-time demonstration
│   ├── test_model.py        # Test script for model evaluation
│   ├── create_scaler.py     # Feature scaler creation
│   ├── test_mic.py          # Microphone test utility
│   └── __init__.py          # Package initialization
├── tests/                    # Unit tests
├── download_dataset.py       # Dataset download script
├── prepare_dataset.py        # Dataset preparation script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
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

## Updates and Best Practices

### 1. **Label Mapping Consistency**
- The label order used for training and inference must match exactly.
- The correct label order is:
  ```python
  keywords = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
  label_map = {k: i for i, k in enumerate(keywords)}
  ```
- Both `src/test_model.py` and `src/live_demo.py` have been updated to use this order.

### 2. **Feature Extraction Parameters**
- Ensure the following parameters are used for both training and inference:
  - `win_ms = 30` (window size in ms)
  - `overlap = 0.25` (25% overlap)
  - `n_mfcc = 13` (number of MFCCs)
  - `max_frames = 100` (time frames, pad/truncate as needed)
- Both test and live demo scripts have been updated to use these values.

### 3. **Audio Preprocessing**
- The training pipeline applies a bandpass filter (300Hz-3000Hz) and normalization before feature extraction.
- The same preprocessing is now applied in both `src/test_model.py` and `src/live_demo.py` for consistent results.

### 4. **Troubleshooting Model Predictions**
- If the model predicts the wrong keyword with high confidence:
  - Double-check label order in all scripts.
  - Ensure feature extraction parameters match training.
  - Confirm that audio preprocessing (bandpass filter and normalization) is applied before MFCC extraction.
  - If issues persist, verify the model file matches the training configuration.

## Running the Test Script

To test the model on a specific audio file:
```bash
python src/test_model.py
```
- The script will print the true label, predicted keyword, and class probabilities.

## Running the Live Demo

To run the real-time keyword spotting demo:
```bash
python src/live_demo.py
```
- Speak one of the supported keywords into your microphone.
- The detected keyword and confidence will be displayed in real time.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- librosa
- sounddevice
- scikit-learn
- matplotlib
- joblib

Install dependencies with:
```bash
pip install -r requirements.txt
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
