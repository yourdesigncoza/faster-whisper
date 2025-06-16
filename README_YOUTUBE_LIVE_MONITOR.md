# YouTube Live Stream Trading Monitor

A comprehensive Python script that combines real-time YouTube live stream transcription with intelligent trading signal detection. This tool continuously monitors YouTube live streams, transcribes audio content, and analyzes it for trading opportunities and signals.

## 🚀 Features

### Core Functionality
- **Continuous Live Stream Transcription**: Real-time audio transcription using Whisper models
- **Incremental Analysis**: Smart content analysis every 3 minutes on new content only
- **Trading Signal Detection**: Advanced keyword and sentiment analysis for trading opportunities
- **Prominent Alerts**: Clear "Boom, Get Ready for a trade!" alerts when signals are detected
- **Rolling Context**: Maintains analysis context across cycles for better accuracy

### Technical Features
- **Robust Error Handling**: Automatic reconnection and retry mechanisms
- **Smart Content Tracking**: Avoids re-analyzing previously processed content
- **Configurable Parameters**: Customizable analysis intervals and model settings
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Signal Strength Scoring**: 0-10 scale for trading signal confidence

## 📋 Requirements

### Dependencies
```bash
pip install faster-whisper numpy yt-dlp openai python-dotenv
```

### System Requirements
- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key (for analysis)

## 🛠️ Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd faster-whisper
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   - Ensure your `.env` file contains your OpenAI API key
   - The script uses the existing configuration system

## 🎯 Usage

### Basic Usage
```bash
python3 youtube_live_monitor.py --url "https://www.youtube.com/watch?v=LIVE_STREAM_ID"
```

### Advanced Usage
```bash
# Custom analysis interval (2 minutes instead of 3)
python3 youtube_live_monitor.py --url "https://youtu.be/LIVE_STREAM_ID" --interval 120

# Different model size
python3 youtube_live_monitor.py --url "https://youtu.be/LIVE_STREAM_ID" --model base

# Custom retry settings
python3 youtube_live_monitor.py --url "https://youtu.be/LIVE_STREAM_ID" --max-retries 15
```

### Interactive Mode
If you don't provide a URL, the script will prompt you:
```bash
python3 youtube_live_monitor.py
```

## ⚙️ Default Configuration

The script uses these optimized defaults:
- **Whisper Model**: `small` (good balance of speed and accuracy)
- **Max Retries**: `10` (robust error recovery)
- **Log Level**: `DEBUG` (comprehensive monitoring)
- **Analysis Interval**: `180 seconds` (3 minutes)
- **Analysis Type**: Full incremental analysis with context

## 🔍 How It Works

### 1. Stream Transcription
- Connects to YouTube live stream using `yt-dlp`
- Processes audio in 30-second chunks
- Transcribes using Whisper model
- Saves to `/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt`

### 2. Incremental Analysis
- Every 3 minutes, analyzes only new transcript content
- Maintains context from previous analysis cycles
- Uses OpenAI for intelligent content analysis
- Focuses specifically on trading-related content

### 3. Trading Signal Detection
- Scans for trading keywords and phrases
- Analyzes trader intent and sentiment
- Calculates signal strength (0-10 scale)
- Generates alerts when trading intent is detected

### 4. Alert System
- **High Confidence (8+ signals)**: "🚨 BOOM! GET READY FOR A TRADE! 🚨 High confidence signal detected!"
- **Strong Signal (5-7 signals)**: "⚡ BOOM! GET READY FOR A TRADE! ⚡ Strong signal detected!"
- **Basic Signal (3-4 signals)**: "📈 Boom, Get Ready for a trade! Trading setup detected."

## 📊 Trading Signal Keywords

### Entry Signals
- entry, enter, buy, sell, long, short, position
- trade setup, setup, signal, breakout, bounce
- support, resistance, trend, reversal

### Intent Phrases (Higher Weight)
- looking for, waiting for, watching, ready to
- preparing, setting up, about to, going to trade
- trade incoming, get ready, here we go, this is it

### Confirmation Words (Highest Weight)
- confirmed, triggered, activated, go, now
- execute, take it, boom, perfect, there it is

## 📁 Output Files

### Transcript File
- **Location**: `/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt`
- **Format**: `[YYYY-MM-DD HH:MM:SS] Transcribed content`
- **Updates**: Real-time as audio is transcribed

### Trading Signal Files
- **Location**: `/home/laudes/zoot/projects/faster-whisper/analysis_results/`
- **Format**: `trading_signal_YYYYMMDD_HHMMSS.json`
- **Content**: Complete analysis results when signals are detected

## 🔧 Troubleshooting

### Common Issues

1. **"No audio data received"**
   - Check if the YouTube stream is actually live
   - Verify the URL is correct and accessible

2. **"OpenAI API key not found"**
   - Ensure your `.env` file contains `OPENAI_API_KEY=your_key_here`
   - Check that the key is valid and has sufficient credits

3. **"FFmpeg not found"**
   - Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu/Debian)
   - Or download from https://ffmpeg.org/

4. **High CPU usage**
   - Consider using a smaller model (`tiny` or `base`)
   - Increase analysis interval to reduce frequency

### Performance Tips

- **For faster processing**: Use `--model tiny`
- **For better accuracy**: Use `--model medium` or `large-v3`
- **For less frequent analysis**: Use `--interval 300` (5 minutes)
- **For more aggressive monitoring**: Use `--interval 60` (1 minute)

## 🛑 Stopping the Monitor

- **Graceful shutdown**: Press `Ctrl+C`
- The script will properly stop all threads and save any pending analysis

## 📈 Example Output

```
🚀 Starting YouTube Live Trading Monitor...
🎵 YouTube URL: https://www.youtube.com/watch?v=example
📁 Transcript file: /home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt
⏱️  Analysis interval: 180s
🤖 Model: small, Max retries: 10
💡 Monitor will analyze new content every 3 minutes for trading signals
----------------------------------------------------------------------
🎵 Starting YouTube transcription...
🔍 Starting analysis worker (interval: 180.0s)
✅ Transcribed: Looking at this setup, we might have a good entry point here
🔍 Analyzing 5 new transcript entries...
📊 Analysis complete: 5 entries, 234 chars

============================================================
🚨 ⚡ BOOM! GET READY FOR A TRADE! ⚡ Strong signal detected! 🚨
Signal Strength: 6/10
Time: 2025-06-16 16:45:23
============================================================
```

This monitor provides a powerful combination of real-time transcription and intelligent analysis, making it an invaluable tool for tracking trading opportunities from live streams.
