import librosa
from kws_system import KWSSystem
import matplotlib.pyplot as plt

def main():
    # Initialize KWS system
    print("Initializing KWS System...")
    kws = KWSSystem()
    
    # Load keyword template
    print("\nLoading keyword template...")
    try:
        keyword_audio, sr = librosa.load('keyword.wav', sr=16000)
        print(f"Keyword audio loaded: {len(keyword_audio)/sr:.2f} seconds")
        
        # Set keyword template
        kws.set_keyword_template(keyword_audio)
        print("Keyword template set successfully")
        
        # Load test audio
        print("\nLoading test audio...")
        test_audio, sr = librosa.load('test_audio.wav', sr=16000)
        print(f"Test audio loaded: {len(test_audio)/sr:.2f} seconds")
        
        # Detect keywords
        print("\nDetecting keywords...")
        detections = kws.detect_keyword(test_audio, threshold=0.7)
        
        if detections:
            print(f"\nFound {len(detections)} keyword occurrences:")
            for i, detection in enumerate(detections, 1):
                print(f"Detection {i}:")
                print(f"  - Time: {detection['position']:.2f} seconds")
                print(f"  - Similarity: {detection['similarity']:.3f}")
        else:
            print("\nNo keywords detected")
        
        # Visualize results
        print("\nGenerating visualization...")
        kws.visualize_detection(test_audio, detections)
        plt.show()
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure you have the following files in the current directory:")
        print("1. keyword.wav - A clean recording of the keyword")
        print("2. test_audio.wav - The audio file to search for the keyword")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 