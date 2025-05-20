"""
Keyword Spotting (KWS) System
A system for detecting keywords in audio streams using template matching.
"""

from .model import KWSSystem
from .data_loader import AudioDataLoader
from .feature_extraction import FeatureExtractor

__all__ = ['KWSSystem', 'AudioDataLoader', 'FeatureExtractor'] 