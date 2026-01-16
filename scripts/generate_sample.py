import os
import argparse
import random
from pydub import AudioSegment

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")

def generate_sample(duration_sec, output_path):
    """Generates a random noise sample for testing (placeholder)."""
    # In a real scenario, you'd record from mic or download.
    # Here we just chop a piece from an existing file if available, or generate silence/noise.
    
    # Try to find a real file to copy from
    existing_files = []
    for root, dirs, files in os.walk(DATASET_DIR):
        for f in files:
            if f.endswith(".wav"):
                existing_files.append(os.path.join(root, f))
    
    if existing_files:
        src = random.choice(existing_files)
        print(f"Slicing from {src}")
        audio = AudioSegment.from_file(src)
        # Take random slice
        if len(audio) > duration_sec * 1000:
            start = random.randint(0, len(audio) - duration_sec * 1000)
            sample = audio[start:start + duration_sec * 1000]
        else:
            sample = audio
    else:
        print("No existing wav files found. Generating silence.")
        sample = AudioSegment.silent(duration=duration_sec * 1000)

    sample.export(output_path, format="wav")
    print(f"Generated sample at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a test audio sample.")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds")
    parser.add_argument("--output", type=str, default="sample.wav", help="Output filename")
    
    args = parser.parse_args()
    generate_sample(args.duration, args.output)
