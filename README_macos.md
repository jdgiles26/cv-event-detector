# CV Event Detector - macOS Compatible Version

This is a macOS-compatible version of the CV Event Detector that runs on Apple Silicon (M1/M2/M3/M4) using Metal Performance Shaders (MPS) for GPU acceleration.

## Features

- ✅ Runs natively on macOS (Intel and Apple Silicon)
- ✅ Uses Metal GPU (MPS) for acceleration on Apple Silicon
- ✅ YOLOv8 object detection
- ✅ IoU-based object tracking
- ✅ Event-based clip generation
- ✅ FastAPI REST API
- ✅ Web UI for easy configuration
- ✅ No NVIDIA dependencies required

## Requirements

- macOS 12.0+ (Monterey or later recommended)
- Python 3.9+
- For best performance: Apple Silicon Mac (M1/M2/M3/M4)

## Quick Start

### 1. Start the Server

```bash
./start_macos.sh
```

Or manually:

```bash
# Install dependencies
pip install -r requirements_macos.txt

# Start the server
python3 main_macos.py
```

### 2. Access the Web UI

Open your browser to: **http://localhost:8000**

### 3. Process a Video

1. Click "Create Pipeline" to create a processing pipeline
2. Enter a video file path (or use the test video generator)
3. Set detection parameters (classes to detect)
4. Click "Process Video"

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/api/pipeline` | POST | Create pipeline |
| `/api/pipeline` | DELETE | Delete pipeline |
| `/api/pipelines` | GET | List pipelines |
| `/api/addstream` | POST | Add video stream |
| `/api/stream` | DELETE | Remove stream |
| `/api/streams` | GET | List streams |
| `/docs` | GET | API Documentation (Swagger) |

## Creating a Test Video

```bash
python3 create_test_video.py --output /tmp/test_video.mp4 --duration 30
```

Then use `/tmp/test_video.mp4` as the video path in the Web UI.

## Detection Parameters

### Detection Prompt
Specify objects to detect separated by dots:
- `person` - detect only people
- `person . car . truck` - detect people, cars, and trucks
- `bicycle . motorcycle . bus` - detect bicycles, motorcycles, and buses

### Confidence Threshold
- Range: 0.1 to 1.0
- Default: 0.3
- Lower values detect more objects but may include false positives

### Minimum Detection Threshold
- Number of objects required to trigger an event
- Default: 3
- Higher values require more objects to be present

## Output

Event clips are saved to `/tmp/via/output/` by default:
- `{stream_name}_{timestamp}.mp4` - Video clip
- `{stream_name}_{timestamp}.json` - Metadata (detections, bounding boxes, etc.)

## Architecture

This macOS version replaces NVIDIA DeepStream components:

| NVIDIA Version | macOS Version |
|---------------|---------------|
| DeepStream Pipeline | OpenCV VideoCapture |
| TensorRT Inference | OpenCV DNN + ONNX Runtime |
| NVTracker | Simple IoU Tracker |
| CUDA | Metal Performance Shaders (MPS) |
| GStreamer | OpenCV VideoWriter |

## Performance

On Apple Silicon Macs:
- Object detection: ~30-60 FPS (depending on resolution)
- GPU acceleration via Metal (MPS)
- Lower power consumption compared to CPU-only

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_FOLDER` | `/tmp/via/output` | Where clips are saved |
| `CLIP_CACHE_PRE_EV_TIME` | `5` | Seconds before event to include |
| `CLIP_CACHE_POST_EV_TIME` | `10` | Seconds after event to include |
| `VSS_SERVER_URL` | `http://localhost:8000/verifyAlert` | Alert verification URL |

## Troubleshooting

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "MPS not available"
- Ensure you're on macOS 12.0+
- The system will fall back to CPU automatically

### Low FPS
- Reduce video resolution
- Increase `frame_skip_interval` to process fewer frames
- Reduce `minimum_detection_threshold` to trigger fewer events

## Differences from NVIDIA Version

1. **Object Detector**: Uses YOLOv8 instead of GroundingDINO (requires Triton)
2. **Tracker**: Simple IoU tracker instead of NVMultiObjectTracker
3. **Inference**: OpenCV DNN instead of TensorRT
4. **No VLM Integration**: This version focuses on CV pipeline only

## License

Same as the original NVIDIA CV Event Detector.
