# CCTV Video Event Detection and Summarization Pipeline

This project implements a multi-stage pipeline for processing CCTV footage to detect and extract significant events (e.g., people, vehicles, animals) into short, compressed video clips. It uses computer vision and deep learning to efficiently summarize long surveillance videos.

## Features

- **4-Stage Funnel Architecture**:
  1. Motion Detection (MOG2 background subtraction)
  2. Object Detection & Tracking (YOLOv8 + ByteTrack)
  3. Activity Analysis (CNN + LSTM for anomaly detection)
  4. Clip Extraction & Encoding (H.265/H.264 with FFmpeg)

- **Fallback Mode**: Works without trained activity model weights using heuristic triggers.
- **GPU Acceleration**: Supports CUDA for faster processing.
- **Validation Tools**: Scripts to check GPU, video validity, and pipeline diagnostics.

## Requirements

- Python 3.8+
- GPU with CUDA (recommended for performance)
- FFmpeg (for video encoding)

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch**:
   - Check GPU: `python check_gpu.py`
   - If GPU available: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
   - If not (CPU-only): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
   - Then edit `config.py`: Set `YOLO_DEVICE = "cuda"` or `"cpu"` based on check.

5. **Download YOLO model** (auto-downloads on first run, or manually):
   - `yolov8n.pt` will be downloaded automatically.

## Configuration

Edit `config.py` to customize:
- Input video path
- Output folder
- Detection thresholds
- Model paths
- Clip durations

## Usage

### Basic Run
```bash
python main.py
```
Processes `input_video.mp4` (from `config.py`) and saves clips to `output/`.

### Custom Input/Output
```bash
python main.py --input my_video.mp4 --clips-folder my_clips
```

### Preview Mode (with GUI)
```bash
python main.py --preview
```
Shows annotated frames in real-time.

### Logging
All logs are saved to `logs/` folder:
- Event logs: `{video_name}_event_log.txt`
- Run outputs: Redirect stdout if needed, e.g., `python main.py > logs/run_log.txt`

## Validation and Testing

### Check GPU
```bash
python check_gpu.py
```
Verifies CUDA availability and GPU details.

### Validate Video
```bash
python check_video.py
```
Checks video properties (resolution, FPS, duration).

### Pipeline Validation
```bash
python validate_pipeline.py
```
Processes first 2 minutes and generates diagnostic report in `logs/validation_report/` with:
- Motion masks
- YOLO detections
- Activity decisions
- Sample clip

### Experimental Co-occurrence Detection
```bash
python test_cooccurrence.py
```
Detects car+bus appearing together, saves to `cooccur_clips/`.

### Download Sample Videos
```bash
python download_videos.py
```
Downloads test videos to `data/`.

## Evaluation

### Performance Metrics
- **FPS**: Processing speed (higher is better).
- **Events Detected**: Number of clips generated.
- **Clip Quality**: Check compressed sizes and content.
- **Accuracy**: Manually verify clips contain relevant events.

### Example Output
From a sample run:
- Input: 70-minute video
- Events: 15 (people/vehicles)
- Processing: ~250 FPS on GPU
- Output: 15 H.265 clips (~0.5-1 MB each)

### Logs Analysis
Check `logs/` for:
- Event timestamps and descriptions
- Processing stats (frames, YOLO calls, etc.)
- Errors (if any)

### Validation Report
Run `validate_pipeline.py` to get:
- Stage-by-stage diagnostics
- Visual proofs (images of detections)
- Summary statistics

## Project Structure

- `main.py`: Main pipeline script
- `config.py`: Configuration settings
- `activity_model.py`: CNN+LSTM model for activity analysis
- `check_*.py`: Utility scripts
- `test_*.py`: Experimental/test scripts
- `validate_pipeline.py`: Validation tool
- `requirements.txt`: Dependencies
- `logs/`: All log files (ignored in Git)
- `output/`: Generated clips (ignored)
- `data/`: Input video files

## Troubleshooting

- **No GPU**: Set `YOLO_DEVICE = "cpu"` in `config.py`
- **Video errors**: Ensure FFmpeg is installed and video is valid
- **Model weights**: `activity_model_weights.pth` is optional; pipeline falls back to heuristics
- **Low FPS**: Check GPU drivers or use CPU mode

## Contributing

1. Test changes with `validate_pipeline.py`
2. Ensure logs are in `logs/` folder
3. Update README for new features

## License

[Add license if applicable]