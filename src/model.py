import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
from .feature_extraction import FeatureExtractor

class KWSSystem:
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 40,
                 n_mfcc: int = 13,
                 hop_length: int = 160,
                 win_length: int = 400,
                 model_type: str = 'cnn'):
        """
        Initialize KWS system with specified parameters.
        
        Args:
            sample_rate: Audio sampling rate
            n_mels: Number of mel bands
            n_mfcc: Number of MFCC coefficients
            hop_length: Samples between frames
            win_length: Window length for STFT
            model_type: Type of model to use ('cnn' or 'lstm')
        """
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            hop_length=hop_length,
            win_length=win_length
        )
        self.model_type = model_type
        self.model = None
        self.labels = ['yes', 'no', 'up', 'down', 'left', 'right', 
                      'on', 'off', 'stop', 'go', 'unknown', 'silence']
    
    def create_model(self, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
        """
        Create and compile the model.
        
        Args:
            input_shape: Shape of input features (time_steps, features, channels)
            
        Returns:
            Compiled Keras model
        """
        if self.model_type == 'cnn':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                                     input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.labels), activation='softmax')
            ])
        else:  # LSTM
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True)
            )(inputs)
            x = tf.keras.layers.Attention()([x, x])
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(len(self.labels), 
                                          activation='softmax')(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, 
             train_data: tf.data.Dataset,
             val_data: tf.data.Dataset,
             epochs: int = 50,
             batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            # Get input shape from first batch
            for x, _ in train_data.take(1):
                input_shape = x.shape[1:]
                break
            self.model = self.create_model(input_shape)
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, 
               audio: np.ndarray,
               threshold: float = 0.7) -> List[Dict]:
        """
        Detect keywords in audio.
        
        Args:
            audio: Input audio signal
            threshold: Detection threshold
            
        Returns:
            List of detections with timestamps and confidence scores
        """
        # Check for voice activity
        if not self.feature_extractor.detect_voice_activity(audio):
            return []
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Reshape for model input
        features = np.expand_dims(features, axis=0)
        if self.model_type == 'cnn':
            features = np.expand_dims(features, axis=-1)
        
        # Get predictions
        predictions = self.model.predict(features)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        if confidence > threshold:
            return [{
                'keyword': self.labels[predicted_class],
                'confidence': float(confidence),
                'timestamp': 0.0  # For real-time processing
            }]
        
        return []
    
    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save(path)
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.model = tf.keras.models.load_model(path)
