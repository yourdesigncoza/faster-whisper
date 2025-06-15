# 🎤 Microphone Transcription Scripts

Real-time speech-to-text transcription using faster-whisper and microphone input.

## 📋 Quick Start

1. **Test your setup**:
   ```bash
   source venv/bin/activate
   python test_microphone.py
   ```

2. **Run a quick demo**:
   ```bash
   python demo_mic_listener.py
   ```

3. **Start transcribing**:
   ```bash
   python simple_mic_listener.py
   ```

## 📁 Files Overview

| File | Purpose | Best For |
|------|---------|----------|
| `test_microphone.py` | Test microphone setup | First-time setup |
| `demo_mic_listener.py` | Quick demo recording | Testing functionality |
| `simple_mic_listener.py` | Easy-to-use listener | General use |
| `mic_listener.py` | Advanced real-time listener | Professional use |
| `MICROPHONE_GUIDE.md` | Detailed usage guide | Learning all features |

## 🚀 Features

### ✅ What's Included

- **Real-time microphone recording** using PyAudio
- **Speech transcription** using faster-whisper
- **Voice Activity Detection** (advanced script)
- **Automatic output to file** with timestamps
- **Multiple model sizes** (tiny to large-v3)
- **GPU acceleration support** (CUDA)
- **Language detection** and manual language setting
- **Configurable recording parameters**
- **Interactive and continuous modes**

### 🎯 Output Format

All transcriptions are saved to `output.txt` (or specified file) with timestamps:

```
[2024-01-15 14:30:25] Hello, this is a test transcription.
[2024-01-15 14:30:45] This is another sentence that was spoken.
```

## 🛠️ Installation

The microphone scripts require additional audio libraries:

```bash
# Activate virtual environment
source venv/bin/activate

# Install audio dependencies (already done in setup)
pip install pyaudio soundfile

# Verify installation
python test_microphone.py
```

## 🎮 Usage Examples

### Simple Interactive Mode

```bash
# Start interactive listener
python simple_mic_listener.py

# Commands in interactive mode:
# - Press Enter: Record 5 seconds
# - Type number + Enter: Record that many seconds
# - Type 'q' + Enter: Quit
```

### Continuous Recording

```bash
# Record continuously in 5-second chunks
python simple_mic_listener.py --mode continuous

# Use 10-second chunks
python simple_mic_listener.py --mode continuous --duration 10
```

### Advanced Real-time Listener

```bash
# Basic real-time transcription
python mic_listener.py

# High-quality with GPU
python mic_listener.py --model medium --device cuda

# Adjust for noisy environment
python mic_listener.py --silence-threshold 0.02 --min-duration 2.0
```

### Custom Output and Models

```bash
# Use different model and output file
python simple_mic_listener.py --model small --output meeting_notes.txt

# Specify language for better accuracy
python mic_listener.py --language en --output english_transcription.txt
```

## ⚙️ Configuration

### Model Sizes (Speed vs Quality)

- **`tiny`** - Fastest, basic quality (~39MB)
- **`base`** - Good balance (~74MB) **[Recommended]**
- **`small`** - Better quality (~244MB)
- **`medium`** - High quality (~769MB)
- **`large-v3`** - Best quality (~1550MB)

### Device Options

- **`cpu`** - Works everywhere, slower
- **`cuda`** - GPU acceleration (requires NVIDIA GPU)

### Compute Types

- **`int8`** - Fastest, good quality **[Recommended for CPU]**
- **`float16`** - Balanced **[Recommended for GPU]**
- **`float32`** - Highest quality, slowest

## 🔧 Troubleshooting

### Common Issues

1. **No audio detected**:
   - Check microphone connection
   - Adjust `--silence-threshold 0.005` (more sensitive)
   - Test with system audio recorder first

2. **ALSA warnings (Linux)**:
   - These are normal and can be ignored
   - They don't affect functionality

3. **Poor transcription quality**:
   - Use larger model: `--model small`
   - Speak clearly and reduce background noise
   - Set language: `--language en`

4. **Performance issues**:
   - Use smaller model: `--model tiny`
   - Use CPU with int8: `--device cpu --compute-type int8`

### Audio Device Issues

List available audio devices:
```bash
python test_microphone.py
```

The test will show all available input/output devices and help identify issues.

## 📊 Performance Tips

### For Real-time Use
- Use `base` or `small` model for good balance
- Enable GPU if available: `--device cuda`
- Set language explicitly: `--language en`

### For High Accuracy
- Use `medium` or `large-v3` model
- Use `float16` compute type on GPU
- Minimize background noise

### For Low-resource Systems
- Use `tiny` model
- Use `int8` compute type
- Increase chunk duration for continuous mode

## 🎯 Use Cases

### Meeting Notes
```bash
python mic_listener.py --model base --language en --output meeting_$(date +%Y%m%d).txt
```

### Language Learning
```bash
python simple_mic_listener.py --model small --language es --output spanish_practice.txt
```

### Voice Memos
```bash
python simple_mic_listener.py --mode continuous --duration 30 --output voice_memos.txt
```

### Accessibility
```bash
python mic_listener.py --model medium --language en --silence-threshold 0.005
```

## 🔍 Testing Your Setup

Always start with the test script:

```bash
python test_microphone.py
```

This will verify:
- ✅ PyAudio installation
- ✅ Microphone recording capability
- ✅ faster-whisper model loading
- 📱 Available audio devices

## 📚 Additional Resources

- **`MICROPHONE_GUIDE.md`** - Comprehensive usage guide
- **`DEV_SETUP.md`** - Development environment setup
- **faster-whisper documentation** - Model details and options

## 🤝 Contributing

To improve the microphone scripts:

1. Test with different microphones and environments
2. Add new features or improve existing ones
3. Update documentation
4. Report issues and suggest improvements

## 📝 Notes

- **Audio Format**: Scripts use 16kHz mono audio (optimal for Whisper)
- **File Format**: Output is UTF-8 text with timestamps
- **Threading**: Advanced script uses separate threads for recording and transcription
- **Memory**: Larger models require more RAM/VRAM
- **Latency**: Real-time transcription has ~1-3 second delay depending on model size

---

🎉 **Ready to start transcribing!** Run `python simple_mic_listener.py` to begin.
