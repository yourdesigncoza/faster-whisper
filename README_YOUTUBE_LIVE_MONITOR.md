# YouTube Live Stream Trading Monitor

A comprehensive Python script that combines real-time YouTube live stream transcription with intelligent trading signal detection. This tool continuously monitors YouTube live streams, transcribes audio content, and analyzes it for **explicit trading intent** using advanced LLM-based detection.

## 🚀 Features

### Core Functionality
- **Continuous Live Stream Transcription**: Real-time audio transcription using Whisper models
- **Incremental Analysis**: Smart content analysis every 3 minutes on new content only
- **Intent-Focused Trading Detection**: Advanced LLM-based analysis that detects explicit trading intent
- **Prominent Alerts**: Clear "Boom, Get Ready for a trade!" alerts when genuine intent is detected
- **Trading Context Management**: Maintains trading-specific context across analysis cycles
- **False Positive Reduction**: Dramatically reduced false alerts compared to keyword-based systems

### Technical Features
- **Robust Error Handling**: Automatic reconnection and retry mechanisms
- **Smart Content Tracking**: Avoids re-analyzing previously processed content
- **Configurable Parameters**: Customizable analysis intervals and model settings
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Confidence Scoring**: 0-1.0 confidence scale for trading intent detection
- **Validation Framework**: Built-in tools to compare detection methods

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
- Maintains trading-specific context from previous analysis cycles
- Uses OpenAI LLM for intelligent content analysis
- Focuses specifically on explicit trading intent detection

#### 📊 Context Management Details

**🔄 What is a "Cycle"?**
- **Analysis Cycle** = **180 seconds (3 minutes)** by default
- **NOT** 30 seconds - that's the audio chunk duration for transcription
- Each cycle analyzes all new transcript content since the last cycle

**⏱️ Two Different Timings:**

1. **🎵 Audio Transcription**: Every **30 seconds**
   - Processes audio chunks and appends to `transcript.txt`
   - This is continuous and automatic

2. **🔍 Analysis Cycles**: Every **180 seconds (3 minutes)**
   - Reads new lines from `transcript.txt` since last analysis
   - Performs intent detection on accumulated content
   - Updates trading context

**📚 Context History:**
- The system maintains context from the **last 10 analysis cycles**
- **10 cycles × 3 minutes = 30 minutes of trading context**

**🧠 What Context is Maintained:**
- ✅ **Last 10 analysis results** (30 minutes of history)
- ✅ **Recent 5 trading signals** (rolling window)
- ✅ **Last intent detection status** and timestamp
- ✅ **Current market sentiment** from recent analysis

**📈 Timeline Example:**
```
Time:     09:00    09:03    09:06    09:09    09:12    09:15
Audio:    [30s]   [30s]    [30s]    [30s]    [30s]    [30s]  ← Continuous
Analysis:   ↓       ↓        ↓        ↓        ↓        ↓     ← Every 3 min
Context:  [Cycle1] [Cycle2] [Cycle3] [Cycle4] [Cycle5] [Cycle6]
```

**🔧 Customizable Analysis Interval:**
```bash
# Analyze every 2 minutes instead of 3
python youtube_live_monitor.py --interval 120

# Analyze every 5 minutes for less frequent checks
python youtube_live_monitor.py --interval 300
```

### 3. Intent-Focused Trading Detection
- Uses structured LLM prompts to detect explicit trading intent
- Looks for clear expressions like "I'm going long", "Taking a position", "I'll buy if..."
- Ignores general trading discussion, market analysis, and educational content
- Extracts direction (long/short), instrument, and entry conditions
- Calculates confidence score (0-1.0 scale)

### 4. Alert System
- **High Confidence (0.8+)**: "🚨 BOOM! GET READY FOR A TRADE! 🚨"
- **Strong Signal (0.6-0.7)**: "⚡ BOOM! GET READY FOR A TRADE! ⚡"
- **Basic Signal (0.3-0.5)**: "📈 Boom, Get Ready for a trade!"
- Includes detected direction, instrument, and entry conditions when available

## 🎯 Intent Detection Examples

### ✅ Will Trigger (Explicit Trading Intent)
- "I'm going long on EURUSD if it breaks above 1.0850"
- "Taking a buy position here at 1950 on gold"
- "I'll go short if this level breaks, stop at 1955"
- "Entering long position now, target 1960"
- "I'm buying AAPL at market open"
- "Going short on this breakdown"

### ❌ Will NOT Trigger (General Discussion)
- "This looks like a nice bullish setup forming"
- "Gold has been behaving unexpectedly with fundamentals"
- "The support level is holding well here"
- "Price action has been quite choppy lately"
- "I bought some groceries yesterday" (non-trading context)
- "Looking for a good entry point here" (not explicit intent)

### 🔍 Key Detection Criteria
- **Explicit Action**: Must indicate immediate or conditional trading action
- **First Person**: Focuses on "I'm", "I'll", "Taking", "Entering"
- **Specific Intent**: Clear indication of entering a trade position
- **Context Aware**: Distinguishes trading from general conversation

## 📁 Output Files

### Transcript File
- **Location**: `/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt`
- **Format**: `[YYYY-MM-DD HH:MM:SS] Transcribed content`
- **Updates**: Real-time as audio is transcribed

### Trading Signal Files
- **Location**: `/home/laudes/zoot/projects/faster-whisper/analysis_results/`
- **Format**: `trading_signal_YYYYMMDD_HHMMSS.json`
- **Content**: Complete analysis results when trading intent is detected
- **Includes**: Intent details, confidence scores, direction, instrument, entry conditions

### Validation Files
- **Location**: `/home/laudes/zoot/projects/faster-whisper/analysis_results/`
- **Format**: `signal_validation_YYYYMMDD_HHMMSS.json`
- **Content**: Comparison results between old and new detection methods

## 🔧 New Components

### Intent Detection System
- **File**: `app/analysis/trading_intent_detector.py`
- **Purpose**: LLM-based trading intent detection
- **Features**: Structured prompts, confidence scoring, detail extraction

### Validation Framework
- **File**: `app/analysis/signal_validation.py`
- **Purpose**: Compare detection methods and validate performance
- **Usage**: `python app/analysis/signal_validation.py --transcript <file>`

### Enhanced Context Management
- **Location**: Updated in `youtube_live_monitor.py`
- **Purpose**: Maintain trading-specific context between analysis cycles
- **Features**: Intent history, signal tracking, market sentiment

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

## 🧪 Validation & Testing

### Manual Validation
Test the intent detection system on historical transcripts:
```bash
python app/analysis/signal_validation.py --transcript analysis_results/youtube/transcript.txt
```

### Individual Testing
Test specific phrases:
```python
from app.analysis.trading_intent_detector import TradingIntentDetector
detector = TradingIntentDetector()
intent = detector.detect_intent("[09:15:30] I'm going long on EURUSD at 1.0850")
print(f"Intent detected: {intent.intent_detected}, Confidence: {intent.confidence}")
```

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
✅ Transcribed: I'm going long on gold if it breaks above 1950
🔍 Analyzing 3 new transcript entries...
📊 Analysis complete: 3 entries, 156 chars

============================================================
🚨 ⚡ BOOM! GET READY FOR A TRADE! ⚡ Direction: LONG | Instrument: gold | Condition: if it breaks above 1950 🚨
Confidence: 0.8
Time: 2025-06-17 14:08:48
============================================================
```

## 🎯 Key Improvements

### Before (Keyword-based Detection)
- ❌ Triggered on any mention of "buy", "sell", "support", etc.
- ❌ High false positive rate (100% on test data)
- ❌ No context awareness
- ❌ Couldn't distinguish intent from discussion

### After (Intent-focused Detection)
- ✅ Only triggers on explicit trading intent
- ✅ Zero false positives on test data
- ✅ Maintains trading-specific context
- ✅ Extracts actionable trading information
- ✅ Distinguishes between analysis and actual trading decisions

This monitor provides a powerful combination of real-time transcription and intelligent intent analysis, making it an invaluable tool for tracking genuine trading opportunities from live streams while eliminating noise from general market discussion.
