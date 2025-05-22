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
  - Pre-trained model with MFCC features (30ms window, 25% overlap)

- **Audio Preprocessing**
  - Bandpass filtering (300Hz - 3000Hz) for speech enhancement
  - Audio normalization for consistent signal levels
  - Voice Activity Detection (VAD) for real-time processing
  - Feature scaling for improved model performance

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
- Feature scaling is applied using a pre-trained scaler for consistent feature ranges.

### 4. **Troubleshooting Model Predictions**
- If the model predicts the wrong keyword with high confidence:
  - Double-check label order in all scripts.
  - Ensure feature extraction parameters match training.
  - Confirm that audio preprocessing (bandpass filter and normalization) is applied before MFCC extraction.
  - Verify that feature scaling is applied correctly.
  - If issues persist, verify the model file matches the training configuration.

## Running the Test Script

To test the model on a specific audio file:
```bash
python src/test_model.py
```
- The script will print the true label, predicted keyword, and class probabilities.
- Uses the pre-trained model from `models/keyword_spotting_mfcc_30ms_25ol.h5`

## Running the Live Demo

To run the real-time keyword spotting demo:
```bash
python src/live_demo.py
```
- Speak one of the supported keywords into your microphone.
- The detected keyword and confidence will be displayed in real time.
- Real-time audio visualization and confidence scores for all keywords.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- librosa
- sounddevice
- scikit-learn
- matplotlib
- joblib
- numpy
- pandas

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Model Architecture

### CNN Model
```python
model = tf.keras.Sequential([
    # First convolutional block
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Classification layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

The CNN model architecture consists of:
- Two convolutional blocks with increasing filters (32 → 64)
- MaxPooling layers after each convolutional block
- A dense layer with 128 units and ReLU activation
- Final softmax layer for classification

### LSTM Model
```python
inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
x = tf.keras.layers.Attention()([x, x])
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

The LSTM model architecture consists of:
- A bidirectional LSTM layer with 128 units
- Self-attention mechanism for capturing temporal dependencies
- Global average pooling to reduce sequence length
- A dense layer with 64 units and ReLU activation
- Dropout layer (0.5) for regularization
- Final softmax layer for classification

## Training

The model training process includes:
1. Feature extraction (MFCC or mel spectrogram)
   - Window size: 30ms
   - Overlap: 25%
   - Number of MFCCs: 13
   - Maximum frames: 100
2. Model training with early stopping
   - Batch size: 32
   - Epochs: 20
   - Learning rate: 0.001
   - Optimizer: Adam
   - Loss: Categorical Cross-entropy
3. Model evaluation on validation set
   - Validation split: 20% of training data
   - Test split: 10% of total data
4. Feature scaling for improved model performance

Note: While data augmentation (time shifting, noise addition) is a common practice in speech recognition, it was not implemented in this version of the model. Future improvements could include:
- Time shifting for temporal invariance
- Noise addition for robustness
- Speed perturbation
- Pitch shifting
- Volume adjustment

## Evaluation

The system is evaluated using:
- Accuracy (achieved ~90% on test set)
- Training and validation loss curves
- Per-class prediction probabilities
- Real-time performance metrics:
  - Inference time per sample
  - Memory usage
  - CPU/GPU utilization

## Real-time Processing

The live demonstration includes:
1. Voice Activity Detection
2. Real-time feature extraction
3. Model inference
4. Confidence thresholding
5. Keyword detection output
6. Real-time visualization

## Best Practices

1. **Data Preparation**
   - Use high-quality audio recordings
   - Apply data augmentation
   - Balance class distribution
   - Include background noise
   - Apply consistent preprocessing

2. **Model Selection**
   - CNN for faster inference
   - LSTM for better accuracy
   - Consider model size constraints
   - Balance accuracy vs. latency
   - Use feature scaling

3. **System Tuning**
   - Adjust detection threshold
   - Optimize feature parameters
   - Fine-tune model architecture
   - Monitor system performance
   - Calibrate feature scaling

## Troubleshooting

1. **Training Issues**
   - Check data quality
   - Verify feature extraction
   - Monitor learning rate
   - Adjust batch size
   - Validate feature scaling

2. **Inference Problems**
   - Verify model loading
   - Check audio input
   - Adjust VAD threshold
   - Monitor system resources
   - Validate feature scaling

3. **Performance Issues**
   - Optimize feature extraction
   - Reduce model complexity
   - Use batch processing
   - Profile system bottlenecks
   - Monitor memory usage




