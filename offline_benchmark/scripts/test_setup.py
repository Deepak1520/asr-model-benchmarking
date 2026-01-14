import os
from faster_whisper import WhisperModel
import whisperx
import torch

# quick check script
TEST_WAV = "dataset/clean_0.wav"

def check_dependencies():
    print("Checking dependencies...")
    
    # Faster Whisper
    try:
        print("Loading Faster-Whisper...")
        m = WhisperModel("tiny", device="cpu", compute_type="int8")
        if os.path.exists(TEST_WAV):
            m.transcribe(TEST_WAV)
        print("Faster-Whisper OK")
    except Exception as e:
        print(f"Faster-Whisper Error: {e}")

    # WhisperX
    try:
        print("Loading WhisperX...")
        try:
            # torch fix
            import torch.serialization
            import omegaconf
            torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
        except: pass
        
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = whisperx.load_model("tiny", dev, compute_type="int8")
        if os.path.exists(TEST_WAV):
             audio = whisperx.load_audio(TEST_WAV)
             m.transcribe(audio)
        print(f"WhisperX OK (device={dev})")
    except Exception as e:
        print(f"WhisperX Error: {e}")

if __name__ == "__main__":
    check_dependencies()
