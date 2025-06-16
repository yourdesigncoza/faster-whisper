# YouTube Transcription Analysis with OpenAI

This directory contains the OpenAI integration for analyzing transcribed YouTube content. The system provides comprehensive analysis capabilities including summarization, key point extraction, sentiment analysis, and custom analysis prompts.

## 🚀 Features

- **📝 Summarization**: Generate concise summaries of transcribed content
- **🔑 Key Point Extraction**: Identify and extract the most important insights
- **😊 Sentiment Analysis**: Analyze emotional tone and sentiment
- **🔍 Custom Analysis**: Use custom prompts for specific analysis needs
- **📦 Batch Processing**: Process multiple transcription files at once
- **⏰ Time Range Analysis**: Analyze specific time periods
- **💾 Result Storage**: Save analysis results in JSON format

## 📁 Project Structure

```
app/
├── analysis/
│   ├── __init__.py
│   ├── openai_client.py      # OpenAI API client wrapper
│   └── analyzer.py           # Main analysis orchestrator
├── utils/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   └── transcription_parser.py  # Transcription file parser
├── analyze_transcription.py  # CLI tool for analysis
└── README.md                 # This file
```

## 🛠️ Installation

1. **Install required dependencies:**
   ```bash
   pip install -r requirements_analysis.txt
   ```

2. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   nano .env
   ```

3. **Required environment variables in `.env`:**
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   DEFAULT_MODEL=gpt-4o-mini
   MAX_TOKENS=2000
   TEMPERATURE=0.3
   ```

## 🎯 Usage

### Command Line Interface

The main CLI tool is `app/analyze_transcription.py`:

```bash
# Full analysis (summary + key points + sentiment)
python app/analyze_transcription.py --full

# Individual analysis types
python app/analyze_transcription.py --summary
python app/analyze_transcription.py --key-points
python app/analyze_transcription.py --sentiment

# Custom analysis with your own prompt
python app/analyze_transcription.py --custom "What trading strategies are mentioned?"

# Analyze specific file
python app/analyze_transcription.py --file custom_output.txt --full

# Batch process multiple files
python app/analyze_transcription.py --batch /path/to/transcription/files --summary

# Don't save results to file
python app/analyze_transcription.py --summary --no-save
```

### Python API

```python
from app.analysis.analyzer import TranscriptionAnalyzer

# Initialize analyzer
analyzer = TranscriptionAnalyzer("output.txt")

# Load transcription data
analyzer.load_transcription()

# Perform full analysis
results = analyzer.analyze_full_transcription()

# Custom analysis
custom_results = analyzer.custom_analysis(
    "What are the main topics discussed in this trading session?"
)

# Time range analysis
from datetime import datetime
start_time = datetime(2024, 1, 15, 14, 0, 0)
end_time = datetime(2024, 1, 15, 15, 0, 0)
range_results = analyzer.analyze_time_range(start_time, end_time)
```

## 📊 Analysis Types

### 1. Summarization
Generates concise summaries that capture the main themes and key points of the transcribed content.

### 2. Key Point Extraction
Identifies and extracts the most important insights, decisions, topics, and actionable items from the content.

### 3. Sentiment Analysis
Analyzes the overall emotional tone, providing:
- Overall sentiment (positive/negative/neutral)
- Confidence score (0-1)
- Key emotions detected
- Detailed explanation

### 4. Custom Analysis
Allows you to specify custom prompts for specific analysis needs, such as:
- "What trading strategies are mentioned?"
- "Summarize the technical analysis discussed"
- "What are the key market predictions?"

## 📁 Output Format

Analysis results are saved as JSON files in the `analysis_results/` directory:

```json
{
  "metadata": {
    "analysis_timestamp": "2024-01-15T14:30:00",
    "transcription_file": "output.txt",
    "statistics": {
      "total_entries": 150,
      "total_words": 2500,
      "total_characters": 15000,
      "time_span": "1:30:00"
    }
  },
  "summary": "Comprehensive summary of the content...",
  "key_points": [
    "First key insight...",
    "Second important point...",
    "Third actionable item..."
  ],
  "sentiment": {
    "sentiment": "positive",
    "confidence": 0.85,
    "key_emotions": ["optimistic", "confident"],
    "explanation": "The content shows positive outlook..."
  }
}
```

## ⚙️ Configuration

Configuration is managed through environment variables and the `app/utils/config.py` file:

- **OPENAI_API_KEY**: Your OpenAI API key (required)
- **DEFAULT_MODEL**: OpenAI model to use (default: gpt-4o-mini)
- **MAX_TOKENS**: Maximum tokens per API call (default: 2000)
- **TEMPERATURE**: Creativity/randomness setting (default: 0.3)
- **BATCH_SIZE**: Number of files to process in batch (default: 10)
- **MAX_CONTENT_LENGTH**: Maximum content length before chunking (default: 8000)

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Integration tests
python tests/test_openai_integration.py --integration

# Unit tests
python tests/test_openai_integration.py --unittest
```

## 🔧 Advanced Features

### Batch Processing
Process multiple transcription files at once:

```bash
python app/analyze_transcription.py --batch /path/to/files --full
```

### Chunking for Large Files
The system automatically handles large transcription files by:
1. Splitting content into manageable chunks
2. Analyzing each chunk separately
3. Combining results intelligently

### Time Range Analysis
Analyze specific time periods within a transcription:

```python
analyzer.analyze_time_range(start_time, end_time)
```

## 🚨 Error Handling

The system includes comprehensive error handling for:
- Missing API keys
- Invalid transcription files
- API rate limits and failures
- Network connectivity issues
- Large file processing

## 💡 Tips for Best Results

1. **Use specific prompts** for custom analysis
2. **Set appropriate temperature** (0.1-0.3 for factual analysis, 0.7+ for creative tasks)
3. **Monitor token usage** for cost optimization
4. **Use batch processing** for multiple files
5. **Save results** for future reference and comparison

## 🔗 Integration with YouTube Listener

This analysis system is designed to work seamlessly with the YouTube listener:

1. Run the YouTube listener to generate transcriptions
2. Use this analysis system to process the transcribed content
3. Get insights and summaries of your YouTube content

Example workflow:
```bash
# Step 1: Transcribe YouTube content
python youtube_listener.py --url "https://youtu.be/example?t=863"

# Step 2: Analyze the transcription
python app/analyze_transcription.py --full

# Step 3: Review results in analysis_results/ directory
```
