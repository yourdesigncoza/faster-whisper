"""
Configuration management for the analysis app.

Handles environment variable loading and configuration settings.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")


class Config:
    """Configuration class for managing environment variables and settings."""

    def __init__(self, env_file: Optional[str] = None, validate_api_key: bool = True):
        """
        Initialize configuration.

        Args:
            env_file: Path to .env file (defaults to .env in project root)
            validate_api_key: Whether to validate API key is present (for testing)
        """
        self.project_root = Path(__file__).parent.parent.parent
        self.validate_api_key = validate_api_key

        if env_file is None:
            env_file = self.project_root / ".env"

        # Load environment variables from .env file if available
        if DOTENV_AVAILABLE and Path(env_file).exists():
            load_dotenv(env_file, override=True)
            print(f"✅ Loaded environment variables from {env_file}")
        elif Path(env_file).exists():
            print(f"⚠️  Found {env_file} but python-dotenv not available")

        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from environment variables."""
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_org_id = os.getenv("OPENAI_ORG_ID")

        # Telegram Configuration
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
        self.telegram_timeout = int(os.getenv("TELEGRAM_TIMEOUT", "10"))

        # Analysis Configuration
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "2000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        
        # File Paths
        self.transcription_file = os.getenv("TRANSCRIPTION_FILE", "output.txt")
        self.analysis_output_dir = os.getenv("ANALYSIS_OUTPUT_DIR", "analysis_results")
        
        # Batch Processing Settings
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.max_content_length = int(os.getenv("MAX_CONTENT_LENGTH", "8000"))
        
        # Validate required settings
        if self.validate_api_key:
            self._validate_config()

    def _validate_config(self):
        """Validate required configuration settings."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your .env file or environment variables.\n"
                "Copy .env.example to .env and add your OpenAI API key."
            )
    
    def get_transcription_path(self) -> Path:
        """Get the full path to the transcription file."""
        transcription_path = Path(self.transcription_file)
        if transcription_path.is_absolute():
            return transcription_path
        else:
            return self.project_root / self.transcription_file
    
    def get_analysis_output_dir(self) -> Path:
        """Get the full path to the analysis output directory."""
        output_path = Path(self.analysis_output_dir)
        if output_path.is_absolute():
            output_dir = output_path
        else:
            output_dir = self.project_root / self.analysis_output_dir
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    def __repr__(self):
        """String representation of configuration (without sensitive data)."""
        return (
            f"Config(\n"
            f"  model={self.default_model},\n"
            f"  max_tokens={self.max_tokens},\n"
            f"  temperature={self.temperature},\n"
            f"  transcription_file={self.transcription_file},\n"
            f"  analysis_output_dir={self.analysis_output_dir},\n"
            f"  batch_size={self.batch_size},\n"
            f"  api_key_set={'Yes' if self.openai_api_key else 'No'}\n"
            f")"
        )


# Global configuration instance
# For testing, we'll create it without validation initially
try:
    config = Config()
except ValueError:
    # If API key validation fails, create config without validation for testing
    config = Config(validate_api_key=False)
