"""
Tests for Enhanced YouTube Listener error handling and reconnection logic.

This module tests the robust error handling features of the enhanced YouTube listener,
including retry mechanisms, automatic reconnection, and graceful degradation.
"""

import pytest
import time
import threading
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

# Add the project root to the path so we can import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from youtube_listener_enhanced import (
    EnhancedYouTubeListener, 
    StreamState, 
    RetryConfig, 
    StreamHealth
)
from app.utils.youtube_config import YouTubeListenerConfig


class TestRetryConfig:
    """Test retry configuration functionality."""
    
    def test_default_retry_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == 5
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True

    def test_custom_retry_config(self):
        """Test custom retry configuration values."""
        config = RetryConfig(
            max_retries=10,
            initial_delay=2.0,
            max_delay=120.0,
            backoff_multiplier=1.5,
            jitter=False
        )
        assert config.max_retries == 10
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_multiplier == 1.5
        assert config.jitter is False


class TestStreamHealth:
    """Test stream health monitoring functionality."""
    
    def test_default_stream_health(self):
        """Test default stream health values."""
        health = StreamHealth()
        assert health.last_audio_time == 0.0
        assert health.consecutive_failures == 0
        assert health.total_reconnects == 0
        assert health.bytes_processed == 0
        assert health.chunks_processed == 0
        assert health.transcriptions_completed == 0


class TestEnhancedYouTubeListener:
    """Test enhanced YouTube listener functionality."""
    
    @pytest.fixture
    def temp_output_file(self):
        """Create a temporary output file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def mock_whisper_model(self):
        """Mock WhisperModel for testing."""
        with patch('youtube_listener_enhanced.WhisperModel') as mock:
            mock_instance = Mock()
            mock_instance.transcribe.return_value = ([], None)
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def listener(self, temp_output_file, mock_whisper_model):
        """Create a test listener instance."""
        retry_config = RetryConfig(max_retries=2, initial_delay=0.1, max_delay=1.0)
        listener = EnhancedYouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test123",
            output_file=temp_output_file,
            retry_config=retry_config,
            log_level="DEBUG"
        )
        return listener

    def test_initialization(self, listener):
        """Test listener initialization."""
        assert listener.youtube_url == "https://www.youtube.com/watch?v=test123"
        assert listener.state == StreamState.INITIALIZING
        assert listener.is_running is False
        assert listener.should_stop is False
        assert isinstance(listener.health, StreamHealth)

    def test_state_transitions(self, listener):
        """Test state transition functionality."""
        assert listener.state == StreamState.INITIALIZING
        
        listener._update_state(StreamState.CONNECTING)
        assert listener.state == StreamState.CONNECTING
        
        listener._update_state(StreamState.STREAMING)
        assert listener.state == StreamState.STREAMING

    def test_health_updates(self, listener):
        """Test health monitoring updates."""
        initial_time = listener.health.last_audio_time
        
        listener._update_health(
            bytes_processed=1024,
            chunks_processed=5,
            transcriptions_completed=2
        )
        
        assert listener.health.bytes_processed == 1024
        assert listener.health.chunks_processed == 5
        assert listener.health.transcriptions_completed == 2
        assert listener.health.last_audio_time > initial_time

    def test_retry_delay_calculation(self, listener):
        """Test retry delay calculation with exponential backoff."""
        # Test exponential backoff
        delay1 = listener._calculate_retry_delay(0)
        delay2 = listener._calculate_retry_delay(1)
        delay3 = listener._calculate_retry_delay(2)
        
        assert delay1 >= 0.1  # initial_delay
        assert delay2 >= delay1  # Should increase
        assert delay3 >= delay2  # Should continue increasing
        assert delay3 <= 1.0  # Should not exceed max_delay

    def test_timestamp_parsing(self, listener):
        """Test YouTube URL timestamp parsing."""
        # Test various timestamp formats
        test_cases = [
            ("https://youtu.be/test?t=863", 863),
            ("https://youtu.be/test?t=863s", 863),
            ("https://youtu.be/test?t=14m23s", 863),  # 14*60 + 23 = 863
            ("https://youtu.be/test?t=1h14m23s", 4463),  # 1*3600 + 14*60 + 23 = 4463
            ("https://youtu.be/test#t=863", 863),
            ("https://youtu.be/test", None),
        ]
        
        for url, expected in test_cases:
            result = listener._parse_timestamp_from_url(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_timestamp_validation(self, listener):
        """Test timestamp validation."""
        assert listener._validate_timestamp(0) is True
        assert listener._validate_timestamp(3600) is True  # 1 hour
        assert listener._validate_timestamp(-1) is False  # Negative
        assert listener._validate_timestamp(90000) is False  # Too large

    def test_speech_content_detection(self, listener):
        """Test speech content detection algorithm."""
        import numpy as np
        
        # Test quiet audio (should be rejected)
        quiet_audio = np.zeros(1000, dtype=np.float32)
        has_speech, reason = listener._has_speech_content(quiet_audio)
        assert has_speech is False
        assert "too quiet" in reason
        
        # Test very loud audio (should be rejected)
        loud_audio = np.ones(1000, dtype=np.float32) * 2.0
        has_speech, reason = listener._has_speech_content(loud_audio)
        assert has_speech is False
        assert "too loud" in reason
        
        # Test reasonable audio (should be accepted)
        reasonable_audio = np.random.normal(0, 0.1, 1000).astype(np.float32)
        has_speech, reason = listener._has_speech_content(reasonable_audio)
        # This might pass or fail depending on the random data, but should not crash

    @patch('subprocess.run')
    def test_youtube_url_validation(self, mock_subprocess, listener):
        """Test YouTube URL validation."""
        # Test valid URLs
        valid_urls = [
            "https://www.youtube.com/watch?v=test123",
            "https://youtube.com/watch?v=test123",
            "https://youtu.be/test123",
            "https://m.youtube.com/watch?v=test123"
        ]
        
        for url in valid_urls:
            listener.youtube_url = url
            try:
                listener._validate_youtube_url()
            except Exception:
                pytest.fail(f"Valid URL rejected: {url}")
        
        # Test invalid URLs
        invalid_urls = [
            "https://example.com/video",
            "not_a_url",
            "https://vimeo.com/123456"
        ]
        
        for url in invalid_urls:
            listener.youtube_url = url
            with pytest.raises(ValueError):
                listener._validate_youtube_url()

    @patch('subprocess.run')
    def test_retry_with_backoff_success(self, mock_subprocess, listener):
        """Test retry mechanism with eventual success."""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = listener._retry_with_backoff(failing_operation, "Test operation")
        assert result == "success"
        assert call_count == 2

    @patch('subprocess.run')
    def test_retry_with_backoff_failure(self, mock_subprocess, listener):
        """Test retry mechanism with persistent failure."""
        def always_failing_operation():
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            listener._retry_with_backoff(always_failing_operation, "Test operation")

    def test_get_status(self, listener):
        """Test status reporting functionality."""
        status = listener.get_status()
        
        assert "state" in status
        assert "is_running" in status
        assert "health" in status
        assert "config" in status
        
        assert status["state"] == StreamState.INITIALIZING.value
        assert status["is_running"] is False
        
        # Test health data structure
        health = status["health"]
        assert "last_audio_time" in health
        assert "consecutive_failures" in health
        assert "total_reconnects" in health
        assert "bytes_processed" in health
        assert "chunks_processed" in health
        assert "transcriptions_completed" in health


class TestYouTubeListenerConfig:
    """Test YouTube listener configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = YouTubeListenerConfig()
        
        assert config.output_file == "output.txt"
        assert config.log_level == "INFO"
        assert config.retry.max_retries == 5
        assert config.monitoring.activity_timeout == 120.0
        assert config.audio.chunk_duration == 30.0
        assert config.transcription.model_size == "base"

    def test_config_to_dict(self):
        """Test configuration serialization to dictionary."""
        config = YouTubeListenerConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "output_file" in config_dict
        assert "retry" in config_dict
        assert "monitoring" in config_dict
        assert "audio" in config_dict
        assert "transcription" in config_dict

    @patch.dict(os.environ, {
        'YOUTUBE_MAX_RETRIES': '10',
        'YOUTUBE_MODEL_SIZE': 'small',
        'YOUTUBE_DEVICE': 'cuda'
    })
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        config = YouTubeListenerConfig()
        
        assert config.retry.max_retries == 10
        assert config.transcription.model_size == "small"
        assert config.transcription.device == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
