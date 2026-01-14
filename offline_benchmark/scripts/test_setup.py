import os
import time
from faster_whisper import WhisperModel
import whisperx
import torch

TEST_FILE = "dataset/clean_0.wav"

def test_faster_whisper():
    print("Testing Faster-Whisper...")
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8") # Use tiny for quick test
        # Note: Benchmark uses large-v2. This confirms library works.
        
        if os.path.exists(TEST_FILE):
             segments, info = model.transcribe(TEST_FILE, beam_size=1)
             print("Faster-Whisper Transcription:", " ".join([s.text for s in segments]).strip())
        else:
             print("Test file not found, skipping transcription.")
        print("Faster-Whisper OK")
    except Exception as e:
        print(f"Faster-Whisper FAILED: {e}")

def test_whisperx():
    print("Testing WhisperX...")
    try:
        try:
            from omegaconf import listconfig
            import torch.serialization
            torch.serialization.add_safe_globals([listconfig.ListConfig])
        except:
             pass
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"WhisperX Device: {device}")
        # load model
        model = whisperx.load_model("tiny", device=device, compute_type="float16" if device=="mps" else "int8")
        
        if os.path.exists(TEST_FILE):
            audio = whisperx.load_audio(TEST_FILE)
            result = model.transcribe(audio, batch_size=4)
            print("WhisperX Transcription:", " ".join([s['text'] for s in result['segments']]).strip())
        print("WhisperX OK")
    except Exception as e:
        print(f"WhisperX FAILED: {e}")

if __name__ == "__main__":
    if not os.path.exists(TEST_FILE):
        print(f"Warning: {TEST_FILE} not found. Ensure dataset is generated.")
    
    test_faster_whisper()
    test_whisperx()
