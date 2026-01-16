import os
import time
import io
import csv
import logging
import shutil
import subprocess
import gc
from typing import Optional

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import jiwer
import psutil
import torch
from pydub import AudioSegment
from pydub.generators import WhiteNoise

# --- Imports with Error Handling ---
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import whisperx
except ImportError:
    whisperx = None

# --- Configuration ---
MODEL_SIZE = "large-v2"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Path Setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "realtime_results.csv")

WHISPER_CPP_DIR = os.path.join(PROJECT_ROOT, "whisper.cpp")
WHISPER_CPP_BINARY = os.path.join(WHISPER_CPP_DIR, "build/bin/whisper-cli")
WHISPER_CPP_MODEL = os.path.join(WHISPER_CPP_DIR, "models/ggml-large-v2.bin")

# Binary path fallback
if not os.path.exists(WHISPER_CPP_BINARY):
    fallback = WHISPER_CPP_BINARY.replace("whisper-cli", "main")
    if os.path.exists(fallback):
        WHISPER_CPP_BINARY = fallback

# Ensure Results Directory Exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(CURRENT_DIR, "templates"))

# --- Model Management ---
class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model_instance = None
        self.load_time = 0.0

    def load_model(self, model_name: str):
        if self.current_model_name == model_name:
            return

        logger.info(f"Switching model to {model_name}...")
        self.unload_model()
        
        t0 = time.time()
        
        if model_name == "Faster-Whisper":
            if not WhisperModel: raise ImportError("Faster-Whisper not installed")
            self.model_instance = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            
        elif model_name == "WhisperX":
            if not whisperx: raise ImportError("WhisperX not installed")
            # Apply safe globals patch for torch 2.6+ if needed
            self._apply_safe_globals()
            self.model_instance = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
            
        elif model_name == "whisper.cpp":
            if not os.path.exists(WHISPER_CPP_BINARY):
                raise FileNotFoundError(f"Binary not found: {WHISPER_CPP_BINARY}")
            self.model_instance = "binary"
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        self.load_time = time.time() - t0
        self.current_model_name = model_name
        logger.info(f"Loaded {model_name} in {self.load_time:.2f}s")
    
    def unload_model(self):
        if self.model_instance:
            del self.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model_instance = None
            self.current_model_name = None

    def _apply_safe_globals(self):
        """Patches torch serialization for WhisperX on newer torch versions."""
        try:
            import torch.serialization
            import omegaconf
            from omegaconf import listconfig, base, dictconfig, nodes
            import typing
            import collections
            import pyannote.audio.core.model
            import pyannote.audio.core.task
            
            torch.serialization.add_safe_globals([
                listconfig.ListConfig, base.ContainerMetadata, dictconfig.DictConfig,
                base.Node, nodes.AnyNode, base.Metadata, typing.Any, list,
                collections.defaultdict, dict, int, set,
                torch.torch_version.TorchVersion,
                pyannote.audio.core.model.Introspection,
                pyannote.audio.core.task.Specifications,
                pyannote.audio.core.task.Problem,
                pyannote.audio.core.task.Resolution
            ])
        except Exception as e:
            logger.warning(f"Could not apply safe globals patch: {e}")

manager = ModelManager()

# --- Helpers ---
def get_current_memory():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def add_noise_to_audio(file_path, snr_db=10):
    try:
        audio = AudioSegment.from_file(file_path)
        noise = WhiteNoise().to_audio_segment(duration=len(audio))
        target_noise_db = audio.dBFS - snr_db
        noise = noise.apply_gain(target_noise_db - noise.dBFS)
        noisy_audio = audio.overlay(noise)
        noisy_audio.export(file_path, format="wav")
        logger.info(f"Injected noise at {snr_db}dB SNR")
    except Exception as e:
        logger.error(f"Error injecting noise: {e}")

def run_faster_whisper(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=5, language="en")
    text = " ".join([s.text for s in segments]).strip()
    return text, info.duration

def run_whisperx(model, audio_path):
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=4, language="en")
    text = " ".join([seg["text"] for seg in result["segments"]]).strip()
    duration = len(audio) / 16000.0
    return text, duration

def run_whisper_cpp(audio_path):
    cmd = [
        WHISPER_CPP_BINARY,
        "-m", WHISPER_CPP_MODEL,
        "-f", audio_path,
        "-nt", "--no-timestamps",
        "-l", "en"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    # Estimate duration
    duration = os.path.getsize(audio_path) / 32000.0
    return stdout.strip(), duration

# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile, 
    reference: str = Form(...), 
    model_name: str = Form(...),
    inject_noise: bool = Form(False)
):
    # 1. Load Model
    try:
        manager.load_model(model_name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # 2. Process Audio File
    temp_wav_path = os.path.join(CURRENT_DIR, f"temp_{int(time.time())}.wav")
    temp_raw_path = os.path.join(CURRENT_DIR, f"temp_in_{int(time.time())}")
    
    try:
        content = await audio.read()
        with open(temp_raw_path, "wb") as f:
            f.write(content)
        
        # Convert to 16kHz Mono WAV
        try:
            audio_segment = AudioSegment.from_file(temp_raw_path)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio_segment.export(temp_wav_path, format="wav")
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            shutil.copy(temp_raw_path, temp_wav_path)

        if inject_noise:
            add_noise_to_audio(temp_wav_path)

        # 3. Inference
        mem_before = get_current_memory()
        t0 = time.time()
        
        if model_name == "Faster-Whisper":
            transcription, duration = run_faster_whisper(manager.model_instance, temp_wav_path)
        elif model_name == "WhisperX":
            transcription, duration = run_whisperx(manager.model_instance, temp_wav_path)
        elif model_name == "whisper.cpp":
            transcription, duration = run_whisper_cpp(temp_wav_path)
        else:
            raise ValueError("Unknown Model")

        latency = time.time() - t0
        mem_after = get_current_memory()
        peak_memory = max(mem_before, mem_after)
        rtf = latency / duration if duration > 0 else 0
        
        # 4. Metrics
        wer = jiwer.wer(reference, transcription)
        cer = jiwer.cer(reference, transcription)
        
        # 5. Log Results
        new_file = not os.path.exists(RESULTS_FILE)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["Timestamp", "Model", "Duration_s", "Latency_s", "RTF", "WER", "CER", "LoadTime_s", "PeakMemory_MB", "Reference", "Transcription"])
            writer.writerow([timestamp, model_name, round(duration, 3), round(latency, 3), round(rtf, 3), round(wer, 4), round(cer, 4), round(manager.load_time, 2), round(peak_memory, 2), reference, transcription])
            
        return JSONResponse({
            "transcription": transcription,
            "metrics": {
                "latency": latency,
                "duration": duration,
                "rtf": rtf,
                "wer": wer,
                "cer": cer,
                "load_time": manager.load_time,
                "peak_memory": peak_memory
            }
        })
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
        
    finally:
        # Cleanup
        for p in [temp_wav_path, temp_raw_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
