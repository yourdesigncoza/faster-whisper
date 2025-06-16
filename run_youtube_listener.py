#!/usr/bin/env python3
"""
YouTube Listener Runner Script

This script provides a convenient way to run the enhanced YouTube listener
from the project root directory. It imports and runs the listener from the
app directory with proper configuration integration.

Usage:
    python run_youtube_listener.py --url "https://youtu.be/VIDEO_ID"
    python run_youtube_listener.py --url "https://youtu.be/VIDEO_ID?t=863" --model small
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Import the YouTube listener from the app directory
    from app.youtube_listener import main, create_listener_from_config
    
    if __name__ == "__main__":
        # Run the main function
        main()
        
except ImportError as e:
    print(f"❌ Error importing YouTube listener: {e}")
    print("Make sure you're running this from the project root directory.")
    print("Required dependencies:")
    print("  - faster-whisper")
    print("  - yt-dlp")
    print("  - numpy")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running YouTube listener: {e}")
    sys.exit(1)
