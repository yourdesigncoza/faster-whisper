# Microphone Listener Guide

This guide explains how to use the microphone listening scripts for real-time speech transcription with faster-whisper.

## 🎯 Overview

We've created three scripts for microphone-based transcription:

1. **`test_microphone.py`** - Test your microphone and audio setup
2. **`simple_mic_listener.py`** - Simple, easy-to-use microphone listener
3. **`mic_listener.py`** - Advanced real-time listener with voice activity detection

## 🧪 Testing Your Setup

First, verify your microphone and audio setup:

```bash
# Activate virtual environment
source venv/bin/activate

# Test microphone setup
python test_microphone.py
```

This will:
- ✅ Test PyAudio installation
- ✅ Test microphone recording (3-second test)
- ✅ Test faster-whisper model loading
- 📱 List all available audio devices

## 🎤 Simple Microphone Listener

The simple listener is perfect for getting started:

### Basic Usage

```bash
# Run with default settings (base model, output.txt)
python simple_mic_listener.py

# Specify model size and output file
python simple_mic_listener.py --model tiny --output my_transcription.txt
```

### Interactive Mode (Default)

In interactive mode, you control when to record:

```
🚀 Simple Microphone Listener
📁 Output file: /path/to/output.txt

Commands:
  Press Enter to record 5 seconds
  Type 'q' and press Enter to quit
  Type a number (e.g., '10') to record for that many seconds

🎤 Press Enter to record (or 'q' to quit): 
```

**Usage:**
- Press **Enter** → Records 5 seconds, then transcribes
- Type **10** and press Enter → Records 10 seconds
- Type **q** and press Enter → Quit

### Continuous Mode

For hands-free operation:

```bash
# Record in 5-second chunks continuously
python simple_mic_listener.py --mode continuous

# Use 10-second chunks
python simple_mic_listener.py --mode continuous --duration 10
```

## 🚀 Advanced Microphone Listener

The advanced listener provides real-time voice activity detection:

### Basic Usage

```bash
# Run with default settings
python mic_listener.py

# Use different model and settings
python mic_listener.py --model small --device cuda --language en
```

### Features

- **Voice Activity Detection**: Automatically detects when you start/stop speaking
- **Real-time Processing**: Transcribes speech as you speak
- **Configurable Thresholds**: Adjust sensitivity for your environment
- **Multi-threading**: Recording and transcription happen simultaneously

### Advanced Options

```bash
# Adjust sensitivity for noisy environments
python mic_listener.py --silence-threshold 0.02

# Set minimum speech duration (avoid transcribing short sounds)
python mic_listener.py --min-duration 2.0

# Force transcription after 20 seconds of continuous speech
python mic_listener.py --max-duration 20.0

# Use GPU acceleration
python mic_listener.py --device cuda --compute-type float16
```

## 📝 Output Format

All scripts save transcriptions to a text file with timestamps:

```
[2024-01-15 14:30:25] Hello, this is a test transcription.
[2024-01-15 14:30:45] This is another sentence that was spoken.
[2024-01-15 14:31:02] The transcription includes timestamps for each segment.
```

## ⚙️ Configuration Options

### Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~39MB | Fastest | Basic | Quick testing |
| `base` | ~74MB | Fast | Good | General use |
| `small` | ~244MB | Medium | Better | Balanced |
| `medium` | ~769MB | Slow | High | High accuracy |
| `large-v3` | ~1550MB | Slowest | Highest | Best quality |

### Device Options

- **`cpu`** - Run on CPU (works everywhere, slower)
- **`cuda`** - Run on GPU (requires NVIDIA GPU with CUDA)

### Compute Types

- **`int8`** - Fastest, lowest memory, good quality
- **`float16`** - Balanced speed and quality (GPU only)
- **`float32`** - Highest quality, slowest

## 🔧 Troubleshooting

### No Audio Detected

1. **Check microphone connection**
2. **Test with system audio recorder first**
3. **Adjust silence threshold**: `--silence-threshold 0.005` (more sensitive)
4. **Check audio permissions** for your terminal/application

### ALSA Warnings (Linux)

The ALSA warnings you might see are normal and don't affect functionality:
```
ALSA lib pcm_dsnoop.c:601:(snd_pcm_dsnoop_open) unable to open slave
```

These can be ignored or suppressed by setting:
```bash
export ALSA_PCM_CARD=default
export ALSA_PCM_DEVICE=0
```

### Poor Transcription Quality

1. **Use a better model**: `--model small` or `--model medium`
2. **Speak clearly** and at normal volume
3. **Reduce background noise**
4. **Set language explicitly**: `--language en`
5. **Use GPU if available**: `--device cuda`

### Performance Issues

1. **Use smaller model**: `--model tiny`
2. **Use CPU with int8**: `--device cpu --compute-type int8`
3. **Increase chunk duration** for continuous mode
4. **Close other applications** to free up resources

## 📋 Quick Reference

### Simple Listener Commands

```bash
# Interactive mode (default)
python simple_mic_listener.py

# Continuous mode
python simple_mic_listener.py --mode continuous --duration 10

# Different model
python simple_mic_listener.py --model tiny --output my_file.txt
```

### Advanced Listener Commands

```bash
# Basic usage
python mic_listener.py

# High quality with GPU
python mic_listener.py --model medium --device cuda --compute-type float16

# Sensitive detection for quiet environments
python mic_listener.py --silence-threshold 0.005 --min-duration 1.0

# Specific language
python mic_listener.py --language en --output english_transcription.txt
```

## 🎯 Best Practices

1. **Start with the simple listener** to get familiar
2. **Test your microphone** with `test_microphone.py` first
3. **Use appropriate model size** for your hardware
4. **Speak clearly** at normal volume
5. **Minimize background noise** when possible
6. **Use GPU acceleration** if available for better performance
7. **Set language explicitly** if you know it for better accuracy

## 🔄 Example Workflow

1. **Test setup**:
   ```bash
   python test_microphone.py
   ```

2. **Quick test with simple listener**:
   ```bash
   python simple_mic_listener.py --model tiny
   ```

3. **Production use with advanced listener**:
   ```bash
   python mic_listener.py --model base --language en --output meeting_notes.txt
   ```

4. **Check output**:
   ```bash
   cat output.txt
   ```

The transcribed text will be continuously appended to your output file with timestamps!
