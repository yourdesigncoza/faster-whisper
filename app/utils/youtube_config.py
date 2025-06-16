"""
Configuration management for Enhanced YouTube Listener.

This module provides configuration management for the enhanced YouTube listener,
integrating with the existing app configuration system and adding specific
settings for error handling, retry behavior, and monitoring.
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class YouTubeRetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class YouTubeMonitoringConfig:
    """Configuration for monitoring and health checks."""
    activity_timeout: float = 120.0  # seconds without activity before warning
    status_report_interval: float = 60.0  # seconds between status reports
    health_check_interval: float = 10.0  # seconds between health checks
    max_consecutive_failures: int = 10  # max failures before giving up


@dataclass
class YouTubeAudioConfig:
    """Configuration for audio processing."""
    chunk_duration: float = 30.0  # seconds per audio chunk
    sample_rate: int = 16000  # audio sample rate
    timeout_no_data: float = 30.0  # timeout if no audio data received
    buffer_size: int = 0  # FFmpeg buffer size (0 = unbuffered)


@dataclass
class YouTubeTranscriptionConfig:
    """Configuration for transcription processing."""
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    vad_filter: bool = True
    word_timestamps: bool = False
    
    # Speech detection thresholds
    min_rms: float = 0.001
    min_amplitude: float = 0.01
    max_amplitude: float = 1.0
    min_zcr: float = 0.005
    max_zcr: float = 0.4


class YouTubeListenerConfig:
    """
    Comprehensive configuration for the Enhanced YouTube Listener.
    
    Integrates with the existing app configuration system and provides
    specific settings for YouTube stream processing.
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize YouTube listener configuration.

        Args:
            env_file: Path to .env file (defaults to .env in project root)
        """
        self.project_root = Path(__file__).parent.parent.parent
        
        if env_file is None:
            env_file = self.project_root / ".env"

        # Load environment variables from .env file if available
        if DOTENV_AVAILABLE and Path(env_file).exists():
            load_dotenv(env_file)
            print(f"✅ Loaded YouTube listener config from {env_file}")
        elif Path(env_file).exists():
            print(f"⚠️  Found {env_file} but python-dotenv not available")

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""
        # Basic settings
        self.output_file = os.getenv("YOUTUBE_OUTPUT_FILE", "/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt")
        self.log_level = os.getenv("YOUTUBE_LOG_LEVEL", "INFO")
        
        # Retry configuration
        self.retry = YouTubeRetryConfig(
            max_retries=int(os.getenv("YOUTUBE_MAX_RETRIES", "5")),
            initial_delay=float(os.getenv("YOUTUBE_INITIAL_DELAY", "1.0")),
            max_delay=float(os.getenv("YOUTUBE_MAX_DELAY", "60.0")),
            backoff_multiplier=float(os.getenv("YOUTUBE_BACKOFF_MULTIPLIER", "2.0")),
            jitter=os.getenv("YOUTUBE_JITTER", "true").lower() == "true"
        )
        
        # Monitoring configuration
        self.monitoring = YouTubeMonitoringConfig(
            activity_timeout=float(os.getenv("YOUTUBE_ACTIVITY_TIMEOUT", "120.0")),
            status_report_interval=float(os.getenv("YOUTUBE_STATUS_INTERVAL", "60.0")),
            health_check_interval=float(os.getenv("YOUTUBE_HEALTH_CHECK_INTERVAL", "10.0")),
            max_consecutive_failures=int(os.getenv("YOUTUBE_MAX_CONSECUTIVE_FAILURES", "10"))
        )
        
        # Audio configuration
        self.audio = YouTubeAudioConfig(
            chunk_duration=float(os.getenv("YOUTUBE_CHUNK_DURATION", "30.0")),
            sample_rate=int(os.getenv("YOUTUBE_SAMPLE_RATE", "16000")),
            timeout_no_data=float(os.getenv("YOUTUBE_TIMEOUT_NO_DATA", "30.0")),
            buffer_size=int(os.getenv("YOUTUBE_BUFFER_SIZE", "0"))
        )
        
        # Transcription configuration
        self.transcription = YouTubeTranscriptionConfig(
            model_size=os.getenv("YOUTUBE_MODEL_SIZE", "base"),
            device=os.getenv("YOUTUBE_DEVICE", "cpu"),
            compute_type=os.getenv("YOUTUBE_COMPUTE_TYPE", "int8"),
            language=os.getenv("YOUTUBE_LANGUAGE"),
            vad_filter=os.getenv("YOUTUBE_VAD_FILTER", "true").lower() == "true",
            word_timestamps=os.getenv("YOUTUBE_WORD_TIMESTAMPS", "false").lower() == "true",
            min_rms=float(os.getenv("YOUTUBE_MIN_RMS", "0.001")),
            min_amplitude=float(os.getenv("YOUTUBE_MIN_AMPLITUDE", "0.01")),
            max_amplitude=float(os.getenv("YOUTUBE_MAX_AMPLITUDE", "1.0")),
            min_zcr=float(os.getenv("YOUTUBE_MIN_ZCR", "0.005")),
            max_zcr=float(os.getenv("YOUTUBE_MAX_ZCR", "0.4"))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "output_file": self.output_file,
            "log_level": self.log_level,
            "retry": asdict(self.retry),
            "monitoring": asdict(self.monitoring),
            "audio": asdict(self.audio),
            "transcription": asdict(self.transcription)
        }

    def __repr__(self):
        """String representation of configuration."""
        return (
            f"YouTubeListenerConfig(\n"
            f"  output_file={self.output_file},\n"
            f"  log_level={self.log_level},\n"
            f"  retry=YouTubeRetryConfig(max_retries={self.retry.max_retries}, "
            f"max_delay={self.retry.max_delay}),\n"
            f"  monitoring=YouTubeMonitoringConfig(activity_timeout={self.monitoring.activity_timeout}),\n"
            f"  audio=YouTubeAudioConfig(chunk_duration={self.audio.chunk_duration}),\n"
            f"  transcription=YouTubeTranscriptionConfig(model_size={self.transcription.model_size}, "
            f"device={self.transcription.device})\n"
            f")"
        )


def create_sample_env_file(output_path: Optional[str] = None) -> str:
    """
    Create a sample .env file with YouTube listener configuration options.
    
    Args:
        output_path: Path to save the sample file (defaults to .env.youtube.sample)
        
    Returns:
        Path to the created sample file
    """
    if output_path is None:
        output_path = ".env.youtube.sample"
    
    sample_content = """# Enhanced YouTube Listener Configuration
# Copy this to .env and modify as needed

# Basic Settings
YOUTUBE_OUTPUT_FILE=/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt
YOUTUBE_LOG_LEVEL=INFO

# Retry Configuration
YOUTUBE_MAX_RETRIES=5
YOUTUBE_INITIAL_DELAY=1.0
YOUTUBE_MAX_DELAY=60.0
YOUTUBE_BACKOFF_MULTIPLIER=2.0
YOUTUBE_JITTER=true

# Monitoring Configuration
YOUTUBE_ACTIVITY_TIMEOUT=120.0
YOUTUBE_STATUS_INTERVAL=60.0
YOUTUBE_HEALTH_CHECK_INTERVAL=10.0
YOUTUBE_MAX_CONSECUTIVE_FAILURES=10

# Audio Processing Configuration
YOUTUBE_CHUNK_DURATION=30.0
YOUTUBE_SAMPLE_RATE=16000
YOUTUBE_TIMEOUT_NO_DATA=30.0
YOUTUBE_BUFFER_SIZE=0

# Transcription Configuration
YOUTUBE_MODEL_SIZE=base
YOUTUBE_DEVICE=cpu
YOUTUBE_COMPUTE_TYPE=int8
YOUTUBE_LANGUAGE=
YOUTUBE_VAD_FILTER=true
YOUTUBE_WORD_TIMESTAMPS=false

# Speech Detection Thresholds
YOUTUBE_MIN_RMS=0.001
YOUTUBE_MIN_AMPLITUDE=0.01
YOUTUBE_MAX_AMPLITUDE=1.0
YOUTUBE_MIN_ZCR=0.005
YOUTUBE_MAX_ZCR=0.4
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_content)
    
    return output_path


# Create the sample file when this module is imported
if __name__ == "__main__":
    # Create sample configuration file
    sample_path = create_sample_env_file()
    print(f"Created sample configuration file: {sample_path}")

    # Test configuration loading
    config = YouTubeListenerConfig()
    print(f"Configuration loaded successfully:")
    print(config)


# Global configuration instance
youtube_config = None

def get_youtube_config(env_file: Optional[str] = None) -> YouTubeListenerConfig:
    """Get or create the global YouTube configuration instance."""
    global youtube_config
    if youtube_config is None:
        youtube_config = YouTubeListenerConfig(env_file)
    return youtube_config
