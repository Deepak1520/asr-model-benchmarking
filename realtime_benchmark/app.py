import os
import time
import io
import csv
import logging
import shutil
import subprocess
import re
import gc
from typing import List

from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import jiwer
import psutil
import torch
from pydub import AudioSegment
from pydub.generators import WhiteNoise

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
COMPUTE_TYPE = "int8"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "realtime_results.csv")
# Assumes whisper.cpp is in the parent directory of realtime_benchmark (i.e., sibling)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WHISPER_CPP_BINARY = os.path.join(PROJECT_ROOT, "whisper.cpp/build/bin/whisper-cli")
WHISPER_CPP_MODEL = os.path.join(PROJECT_ROOT, "whisper.cpp/models/ggml-large-v2.bin")

# Helper to find binary if strictly named differently
if not os.path.exists(WHISPER_CPP_BINARY):
    fallback = WHISPER_CPP_BINARY.replace("whisper-cli", "main")
    if os.path.exists(fallback):
        WHISPER_CPP_BINARY = fallback

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- Model Manager ---
class ModelManager:
    def __init__(self):
        self.current_model_name = None
        self.model_instance = None
        self.load_time = 0.0

    def load_model(self, model_name):
        if self.current_model_name == model_name:
            return # Already loaded

        print(f"Switching model to {model_name}...")
        self.unload_model() # Free RAM
        
        t0 = time.time()
        
        if model_name == "Faster-Whisper":
            self.model_instance = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            
        elif model_name == "WhisperX":
            # Patches for torch 2.6+ Pickle Security
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
                print(f"Warning: Could not apply safe globals patch: {e}")
            
            self.model_instance = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
            
        elif model_name == "whisper.cpp":
            # No Python object to load, just check binary
            if not os.path.exists(WHISPER_CPP_BINARY):
                raise FileNotFoundError(f"Binary not found: {WHISPER_CPP_BINARY}")
            self.model_instance = "binary"
            
        self.load_time = time.time() - t0
        self.current_model_name = model_name
        print(f"Loaded {model_name} in {self.load_time:.2f}s")
    
    def unload_model(self):
        if self.model_instance:
            del self.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model_instance = None
            self.current_model_name = None

manager = ModelManager()

# --- Helpers ---
def get_current_memory():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def add_noise_to_audio(file_path, snr_db=10):
    """Injects white noise into the audio file at the specified SNR."""
    try:
        audio = AudioSegment.from_file(file_path)
        # Generate noise
        noise = WhiteNoise().to_audio_segment(duration=len(audio))
        
        # Calculate target noise level
        target_noise_db = audio.dBFS - snr_db
        noise = noise.apply_gain(target_noise_db - noise.dBFS)
        
        # Overlay
        noisy_audio = audio.overlay(noise)
        noisy_audio.export(file_path, format="wav")
        print(f"Injected noise at {snr_db}dB SNR")
    except Exception as e:
        print(f"Error injecting noise: {e}")

def run_faster_whisper(model, audio_bytes=None, file_path=None):
    if file_path:
        # Force English
        segments, info = model.transcribe(file_path, beam_size=5, language="en")
    else:
        binary_stream = io.BytesIO(audio_bytes)
        segments, info = model.transcribe(binary_stream, beam_size=5, language="en")
    
    text = " ".join([s.text for s in segments]).strip()
    return text, info.duration

def run_whisperx(model, audio_path):
    # WhisperX needs a file path
    audio = whisperx.load_audio(audio_path)
    # Force English
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
        "-l", "en"  # Force English
    ]
    
    # Run subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate()
    
    transcription = stdout.strip()
    
    # Estimate duration from file size (16k 16bit mono = 32000 bytes/s)
    duration = os.path.getsize(audio_path) / 32000.0
    
    return transcription, duration


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
    
    # 1. Load Model (Lazy)
    try:
        manager.load_model(model_name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # 2. Save Audio to Temp File
    temp_filename = f"temp_{int(time.time())}.wav"
    temp_path = os.path.join(BASE_DIR, temp_filename)
    
    content = await audio.read()
    
    # Save raw input first to help ffmpeg probe format (pipe input can be tricky)
    temp_in_filename = f"temp_in_{int(time.time())}" # No extension, let ffmpeg guess
    temp_in_path = os.path.join(BASE_DIR, temp_in_filename)
    
    with open(temp_in_path, "wb") as f:
        f.write(content)
    
    # Convert incoming audio to 16kHz Mono WAV for whisper.cpp compatibility
    try:
        audio_segment = AudioSegment.from_file(temp_in_path)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio_segment.export(temp_path, format="wav")
    except Exception as e:
        print(f"Audio Conversion Error: {e}")
        # Fallback: copy raw input if conversion fails
        shutil.copy(temp_in_path, temp_path)
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        
    try:
        # 2b. Inject Noise if requested
        if inject_noise:
            add_noise_to_audio(temp_path, snr_db=10)

        # Measure Inference
        mem_before = get_current_memory()
        t0 = time.time()
        
        if model_name == "Faster-Whisper":
             transcription, duration = run_faster_whisper(manager.model_instance, file_path=temp_path)
        elif model_name == "WhisperX":
            transcription, duration = run_whisperx(manager.model_instance, temp_path)
        elif model_name == "whisper.cpp":
            transcription, duration = run_whisper_cpp(temp_path)
        else:
            return JSONResponse({"error": "Unknown model"}, status_code=400)

        t1 = time.time()
        mem_after = get_current_memory()
        peak_memory = max(mem_before, mem_after)
        
        latency = t1 - t0
        rtf = latency / duration if duration > 0 else 0
        
        # Metrics
        ref_norm = jiwer.RemovePunctuation()(reference.lower())
        hyp_norm = jiwer.RemovePunctuation()(transcription.lower())
        wer = jiwer.wer(reference, transcription)
        cer = jiwer.cer(reference, transcription)
        
        # CSV Logging
        if not os.path.exists(RESULTS_FILE):
             with open(RESULTS_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Timestamp", "Model", "Duration_s", "Latency_s", "RTF", "WER", "CER", "LoadTime_s", "PeakMemory_MB", "Reference", "Transcription"])

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(RESULTS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([timestamp, model_name, round(duration, 3), round(latency, 3), round(rtf, 3), round(wer, 4), round(cer, 4), round(manager.load_time, 2), round(peak_memory, 2), reference, transcription])
            
        logger.info(f"Processed with {model_name}: Latency={latency:.2f}s, RTF={rtf:.2f}")

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

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
