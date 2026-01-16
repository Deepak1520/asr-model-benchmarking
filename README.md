# ASR Model Benchmarking

A comprehensive benchmarking suite for comparing Automatic Speech Recognition (ASR) models on Apple Silicon (M3).

## Supported Models
- **Faster-Whisper**: Optimized Transformer implementation.
- **WhisperX**: Features accurate timestamping and diarization.
- **Whisper.cpp**: High-performance C++ inference (CoreML/Metal support).

## Project Structure
```
.
├── src/                # Source code
│   ├── offline/        # Offline benchmarking scripts
│   └── realtime/       # Real-time FastAPI demo
├── scripts/            # Utility scripts
├── data/               # Datasets and Results
│   ├── dataset/        # Audio files (Clean, Noisy, Accented)
│   └── results/        # Benchmark CSVs and Graphs
└── requirements.txt    # Python dependencies
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `whisperx` may require installation from GitHub.*

2. **Whisper.cpp Setup**
   Clone and build `whisper.cpp` in the project root:
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   ./models/download-ggml-model.sh large-v2
   ```

## Usage

### Offline Benchmarking
Run benchmarks for individual models:
```bash
python src/offline/benchmark_fw.py   # Run Faster-Whisper
python src/offline/benchmark_cpp.py  # Run Whisper.cpp
```

Generate graphs and summary tables:
```bash
python src/offline/visualize.py
```
Results will be saved to `data/results/`.

### Real-time Demo
Launch the web interface:
```bash
uvicorn src.realtime.app:app --reload
```
Open `http://127.0.0.1:8000` to transcribe audio files with real-time metrics.

## License
MIT
