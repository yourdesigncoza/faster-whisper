# Enhanced YouTube Listener

A robust YouTube stream transcription system with advanced error handling, automatic reconnection, and comprehensive monitoring capabilities.

## 🚀 Features

### Core Functionality
- **Real-time YouTube stream transcription** using faster-whisper
- **Timestamp seeking support** for starting at specific video positions
- **Automatic speech detection** to skip non-speech audio segments
- **Continuous output** to timestamped text files

### Enhanced Error Handling
- **Automatic reconnection** with exponential backoff
- **Configurable retry behavior** with jitter and maximum delays
- **Graceful degradation** when encountering temporary issues
- **Comprehensive logging** with multiple log levels
- **Stream health monitoring** with real-time status reporting

### Monitoring & Observability
- **Real-time status reporting** every 60 seconds
- **Health metrics tracking** (bytes processed, reconnections, failures)
- **Activity monitoring** with timeout detection
- **Thread health monitoring** to detect worker failures

## 📋 Requirements

```bash
pip install faster-whisper yt-dlp numpy python-dotenv
```

### System Dependencies
- **FFmpeg** - for audio processing
- **yt-dlp** - for YouTube stream extraction

## 🔧 Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_youtube.txt
   ```

2. **Install system dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

3. **Install yt-dlp:**
   ```bash
   pip install yt-dlp
   ```

## 🚀 Quick Start

### Basic Usage

```bash
python youtube_listener_enhanced.py --url "https://youtu.be/VIDEO_ID"
```

### With Timestamp Seeking

```bash
python youtube_listener_enhanced.py --url "https://youtu.be/VIDEO_ID?t=863"
```

### Advanced Configuration

```bash
python youtube_listener_enhanced.py \
  --url "https://youtu.be/VIDEO_ID" \
  --model small \
  --device cuda \
  --max-retries 10 \
  --log-level DEBUG \
  --output transcription.txt
```

## ⚙️ Configuration

### Command Line Options

#### Basic Options
- `--url` - YouTube URL (supports timestamp parameters)
- `--model` - Whisper model size (tiny, base, small, medium, large-v3)
- `--device` - Device for inference (cpu, cuda)
- `--compute-type` - Compute type (int8, float16, float32)
- `--output` - Output file path (default: output.txt)
- `--language` - Language code (e.g., 'en', 'es', 'fr')
- `--chunk-duration` - Audio chunk duration in seconds (default: 30.0)

#### Error Handling Options
- `--max-retries` - Maximum retry attempts (default: 5)
- `--initial-delay` - Initial retry delay in seconds (default: 1.0)
- `--max-delay` - Maximum retry delay in seconds (default: 60.0)
- `--backoff-multiplier` - Backoff multiplier for delays (default: 2.0)
- `--no-jitter` - Disable jitter in retry delays
- `--log-level` - Logging level (DEBUG, INFO, WARNING, ERROR)

### Environment Configuration

Create a `.env` file for persistent configuration:

```bash
# Copy the sample configuration
cp .env.youtube.sample .env
```

Example `.env` configuration:

```env
# Basic Settings
YOUTUBE_OUTPUT_FILE=transcription.txt
YOUTUBE_LOG_LEVEL=INFO

# Retry Configuration
YOUTUBE_MAX_RETRIES=10
YOUTUBE_INITIAL_DELAY=2.0
YOUTUBE_MAX_DELAY=120.0
YOUTUBE_BACKOFF_MULTIPLIER=1.5

# Monitoring Configuration
YOUTUBE_ACTIVITY_TIMEOUT=180.0
YOUTUBE_STATUS_INTERVAL=30.0

# Transcription Configuration
YOUTUBE_MODEL_SIZE=small
YOUTUBE_DEVICE=cuda
YOUTUBE_COMPUTE_TYPE=float16
```

## 🔍 Monitoring & Status

### Real-time Status Reports

The enhanced listener provides regular status updates:

```
📊 Status Report - State: streaming | Chunks: 45 | Transcriptions: 23 | Reconnects: 2 | Failures: 0
```

### Health Metrics

- **Chunks Processed** - Number of audio chunks processed
- **Transcriptions Completed** - Number of successful transcriptions
- **Total Reconnects** - Number of automatic reconnections
- **Consecutive Failures** - Current failure streak
- **Bytes Processed** - Total audio data processed

### Stream States

- `initializing` - Starting up and loading model
- `connecting` - Establishing connection to YouTube stream
- `streaming` - Actively processing audio and transcribing
- `reconnecting` - Attempting to reconnect after failure
- `stopping` - Gracefully shutting down
- `stopped` - Completely stopped
- `error` - Unrecoverable error state

## 🛠️ Error Handling

### Automatic Recovery

The enhanced listener automatically handles:

- **Network interruptions** - Reconnects with exponential backoff
- **Stream timeouts** - Detects and recovers from stalled streams
- **YouTube stream changes** - Adapts to stream URL changes
- **FFmpeg process failures** - Restarts audio processing
- **Temporary transcription errors** - Continues processing other chunks

### Retry Strategy

1. **Exponential Backoff** - Delays increase exponentially (1s, 2s, 4s, 8s...)
2. **Jitter** - Random variation to prevent thundering herd
3. **Maximum Delay** - Caps retry delays at configurable maximum
4. **Maximum Retries** - Gives up after configured number of attempts

### Graceful Degradation

- **Speech Detection** - Skips non-speech audio to save processing
- **Quality Filtering** - Filters out low-quality transcriptions
- **Resource Management** - Monitors memory and CPU usage
- **Clean Shutdown** - Handles Ctrl+C gracefully

## 📊 Comparison with Original

| Feature | Original | Enhanced |
|---------|----------|----------|
| Error Handling | Basic | Comprehensive |
| Reconnection | Manual | Automatic |
| Monitoring | None | Real-time |
| Configuration | Command-line only | Environment + CLI |
| Logging | Print statements | Structured logging |
| Recovery | Fails on error | Automatic retry |
| Status Reporting | None | Detailed metrics |
| Thread Safety | Basic | Full thread safety |

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
python -m pytest tests/test_youtube_enhanced.py -v

# Run specific test categories
python -m pytest tests/test_youtube_enhanced.py::TestRetryConfig -v
python -m pytest tests/test_youtube_enhanced.py::TestEnhancedYouTubeListener -v
```

## 🔧 Troubleshooting

### Common Issues

1. **"yt-dlp error"**
   - Ensure yt-dlp is installed: `pip install yt-dlp`
   - Update yt-dlp: `pip install --upgrade yt-dlp`

2. **"FFmpeg not found"**
   - Install FFmpeg system package
   - Ensure FFmpeg is in system PATH

3. **"Model loading failed"**
   - Check available memory (models require 1-8GB RAM)
   - Try smaller model size (tiny, base instead of large)
   - For CUDA: verify CUDA installation

4. **"Stream keeps disconnecting"**
   - Check network stability
   - Increase retry limits: `--max-retries 20`
   - Increase delays: `--max-delay 300`

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
python youtube_listener_enhanced.py \
  --url "YOUR_URL" \
  --log-level DEBUG \
  --max-retries 20
```

## 📝 Output Format

Transcriptions are saved with timestamps:

```
[2024-01-15 14:30:25] Welcome to this presentation about machine learning.
[2024-01-15 14:30:55] Today we'll cover the basics of neural networks.
[2024-01-15 14:31:20] Let's start with the fundamental concepts.
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the same terms as the faster-whisper project.
