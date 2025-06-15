# Development Environment Setup

This document describes how to set up the development environment for the faster-whisper project.

## Prerequisites

- Python 3.9 or higher (currently using Python 3.10.12)
- Git
- Virtual environment support

## Quick Setup

1. **Clone the repository** (already done):
   ```bash
   git clone https://github.com/yourdesigncoza/faster-whisper.git
   cd faster-whisper
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Verify Installation

Run the installation test:
```bash
python test_installation.py
```

This will verify that all core modules can be imported and basic functionality works.

## Development Tools

We've included a `dev_tools.py` script to help with common development tasks:

### Available Commands

- **Format check**: `python dev_tools.py format`
  - Checks code formatting with black and import sorting with isort
  
- **Lint**: `python dev_tools.py lint`
  - Runs flake8 linting on the codebase
  
- **Test**: `python dev_tools.py test`
  - Runs the full test suite with pytest
  
- **Quick test**: `python dev_tools.py quick-test`
  - Runs a subset of fast tests for quick feedback
  
- **Check all**: `python dev_tools.py check-all`
  - Runs formatting, linting, and tests in sequence
  
- **Fix formatting**: `python dev_tools.py fix-format`
  - Automatically fixes code formatting issues

### Example Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick tests during development
python dev_tools.py quick-test

# Check everything before committing
python dev_tools.py check-all

# Fix formatting issues
python dev_tools.py fix-format
```

## Manual Development Commands

If you prefer to run tools manually:

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tokenizer.py -v

# Run with coverage
python -m pytest tests/ --cov=faster_whisper
```

### Code Quality
```bash
# Format code
black faster_whisper/ tests/
isort faster_whisper/ tests/

# Check formatting
black faster_whisper/ tests/ --check --diff
isort faster_whisper/ tests/ --check-only --diff

# Lint code
flake8 faster_whisper/ tests/
```

## Project Structure

```
faster-whisper/
├── faster_whisper/          # Main package
│   ├── __init__.py
│   ├── transcribe.py        # Core transcription logic
│   ├── audio.py            # Audio processing
│   ├── tokenizer.py        # Text tokenization
│   ├── feature_extractor.py # Feature extraction
│   ├── utils.py            # Utility functions
│   ├── vad.py              # Voice Activity Detection
│   └── version.py          # Version information
├── tests/                  # Test files
│   ├── test_transcribe.py
│   ├── test_tokenizer.py
│   └── test_utils.py
├── benchmark/              # Performance benchmarks
├── docker/                 # Docker configuration
├── requirements.txt        # Production dependencies
├── setup.py               # Package setup
└── venv/                  # Virtual environment (created)
```

## Dependencies

### Core Dependencies
- `ctranslate2>=4.0,<5` - Fast inference engine
- `huggingface_hub>=0.13` - Model downloading
- `tokenizers>=0.13,<1` - Text tokenization
- `onnxruntime>=1.14,<2` - ONNX runtime
- `av>=11` - Audio/video processing
- `tqdm` - Progress bars

### Development Dependencies
- `black==23.*` - Code formatting
- `flake8==6.*` - Code linting
- `isort==5.*` - Import sorting
- `pytest==7.*` - Testing framework

## Tips

1. **Always activate the virtual environment** before working:
   ```bash
   source venv/bin/activate
   ```

2. **Run tests frequently** during development:
   ```bash
   python dev_tools.py quick-test
   ```

3. **Check code quality** before committing:
   ```bash
   python dev_tools.py check-all
   ```

4. **Keep dependencies up to date** but test thoroughly after updates.

## Troubleshooting

### Import Errors
If you get import errors, make sure:
- Virtual environment is activated
- Package is installed in development mode: `pip install -e ".[dev]"`

### Test Failures
If tests fail:
- Check that all dependencies are installed
- Ensure you're in the correct directory
- Try running individual test files to isolate issues

### GPU Support
For GPU support, you may need additional NVIDIA libraries:
- cuBLAS for CUDA 12
- cuDNN 9 for CUDA 12

See the main README.md for detailed GPU setup instructions.
