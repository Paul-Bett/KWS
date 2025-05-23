import sounddevice as sd
import numpy as np

print("Recording 2 seconds of audio from default microphone...")
audio = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype=np.float32)
sd.wait()
print("Recorded shape:", audio.shape) 