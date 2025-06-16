#!/usr/bin/env python3
"""
Demo script to showcase the enhanced YouTube listener features.

This script demonstrates the key improvements in the enhanced YouTube listener
compared to the original version, including error handling, monitoring, and
configuration capabilities.
"""

import time
import tempfile
from youtube_listener_enhanced import (
    EnhancedYouTubeListener, 
    RetryConfig, 
    StreamState,
    StreamHealth
)
from app.utils.youtube_config import YouTubeListenerConfig, create_sample_env_file


def demo_retry_configuration():
    """Demonstrate retry configuration capabilities."""
    print("🔧 Retry Configuration Demo")
    print("=" * 50)
    
    # Default configuration
    default_config = RetryConfig()
    print(f"Default: max_retries={default_config.max_retries}, "
          f"initial_delay={default_config.initial_delay}s, "
          f"max_delay={default_config.max_delay}s")
    
    # Custom configuration for aggressive retries
    aggressive_config = RetryConfig(
        max_retries=20,
        initial_delay=0.5,
        max_delay=300.0,
        backoff_multiplier=1.5,
        jitter=True
    )
    print(f"Aggressive: max_retries={aggressive_config.max_retries}, "
          f"initial_delay={aggressive_config.initial_delay}s, "
          f"max_delay={aggressive_config.max_delay}s")
    
    # Conservative configuration for stable networks
    conservative_config = RetryConfig(
        max_retries=3,
        initial_delay=5.0,
        max_delay=60.0,
        backoff_multiplier=2.0,
        jitter=False
    )
    print(f"Conservative: max_retries={conservative_config.max_retries}, "
          f"initial_delay={conservative_config.initial_delay}s, "
          f"max_delay={conservative_config.max_delay}s")
    print()


def demo_health_monitoring():
    """Demonstrate health monitoring capabilities."""
    print("📊 Health Monitoring Demo")
    print("=" * 50)
    
    health = StreamHealth()
    print(f"Initial health: {health}")
    
    # Simulate some activity
    health.bytes_processed = 1024 * 1024  # 1MB
    health.chunks_processed = 50
    health.transcriptions_completed = 25
    health.total_reconnects = 2
    health.consecutive_failures = 0
    health.last_audio_time = time.time()
    
    print(f"After activity: {health}")
    print()


def demo_timestamp_parsing():
    """Demonstrate timestamp parsing capabilities."""
    print("🕐 Timestamp Parsing Demo")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        listener = EnhancedYouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test",
            output_file=f.name
        )
        
        test_urls = [
            "https://youtu.be/test?t=863",
            "https://youtu.be/test?t=863s", 
            "https://youtu.be/test?t=14m23s",
            "https://youtu.be/test?t=1h14m23s",
            "https://youtu.be/test#t=863",
            "https://youtu.be/test",  # No timestamp
        ]
        
        for url in test_urls:
            timestamp = listener._parse_timestamp_from_url(url)
            if timestamp:
                hours = timestamp // 3600
                minutes = (timestamp % 3600) // 60
                seconds = timestamp % 60
                formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
                print(f"URL: {url}")
                print(f"  → {timestamp}s ({formatted})")
            else:
                print(f"URL: {url}")
                print(f"  → No timestamp")
    print()


def demo_state_management():
    """Demonstrate state management capabilities."""
    print("🔄 State Management Demo")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        listener = EnhancedYouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test",
            output_file=f.name
        )
        
        print(f"Initial state: {listener.state.value}")
        
        # Simulate state transitions
        states = [
            StreamState.CONNECTING,
            StreamState.STREAMING,
            StreamState.RECONNECTING,
            StreamState.STREAMING,
            StreamState.STOPPING,
            StreamState.STOPPED
        ]
        
        for state in states:
            listener._update_state(state)
            print(f"Transitioned to: {state.value}")
            time.sleep(0.1)  # Small delay for demonstration
    print()


def demo_configuration_system():
    """Demonstrate configuration system capabilities."""
    print("⚙️ Configuration System Demo")
    print("=" * 50)
    
    # Create sample configuration file
    sample_path = create_sample_env_file("demo_config.env")
    print(f"Created sample config: {sample_path}")
    
    # Load configuration
    config = YouTubeListenerConfig()
    print(f"Loaded configuration:")
    print(f"  Output file: {config.output_file}")
    print(f"  Log level: {config.log_level}")
    print(f"  Max retries: {config.retry.max_retries}")
    print(f"  Model size: {config.transcription.model_size}")
    print(f"  Device: {config.transcription.device}")
    print(f"  Chunk duration: {config.audio.chunk_duration}s")
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print(f"Configuration sections: {list(config_dict.keys())}")
    print()


def demo_error_handling():
    """Demonstrate error handling capabilities."""
    print("🛡️ Error Handling Demo")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        # Create listener with aggressive retry settings
        retry_config = RetryConfig(max_retries=3, initial_delay=0.1, max_delay=1.0)
        listener = EnhancedYouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test",
            output_file=f.name,
            retry_config=retry_config,
            log_level="INFO"
        )
        
        # Test retry delay calculation
        print("Retry delay progression:")
        for attempt in range(5):
            delay = listener._calculate_retry_delay(attempt)
            print(f"  Attempt {attempt + 1}: {delay:.2f}s")
        
        # Test URL validation
        print("\nURL validation tests:")
        valid_urls = [
            "https://www.youtube.com/watch?v=test123",
            "https://youtu.be/test123",
        ]
        
        invalid_urls = [
            "https://example.com/video",
            "not_a_url"
        ]
        
        for url in valid_urls:
            listener.youtube_url = url
            try:
                listener._validate_youtube_url()
                print(f"  ✅ Valid: {url}")
            except:
                print(f"  ❌ Invalid: {url}")
        
        for url in invalid_urls:
            listener.youtube_url = url
            try:
                listener._validate_youtube_url()
                print(f"  ✅ Valid: {url}")
            except:
                print(f"  ❌ Invalid: {url}")
    print()


def demo_status_reporting():
    """Demonstrate status reporting capabilities."""
    print("📈 Status Reporting Demo")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(suffix='.txt') as f:
        listener = EnhancedYouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test",
            output_file=f.name
        )
        
        # Get initial status
        status = listener.get_status()
        print("Initial status:")
        print(f"  State: {status['state']}")
        print(f"  Running: {status['is_running']}")
        print(f"  Health: {status['health']}")
        
        # Simulate some activity
        listener._update_health(
            bytes_processed=2048,
            chunks_processed=10,
            transcriptions_completed=5
        )
        
        # Get updated status
        status = listener.get_status()
        print("\nAfter simulated activity:")
        print(f"  Chunks processed: {status['health']['chunks_processed']}")
        print(f"  Transcriptions: {status['health']['transcriptions_completed']}")
        print(f"  Bytes processed: {status['health']['bytes_processed']}")
    print()


def main():
    """Run all demonstration functions."""
    print("🎉 Enhanced YouTube Listener Feature Demo")
    print("=" * 60)
    print()
    
    demos = [
        demo_retry_configuration,
        demo_health_monitoring,
        demo_timestamp_parsing,
        demo_state_management,
        demo_configuration_system,
        demo_error_handling,
        demo_status_reporting,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print()
    
    print("🎊 Demo completed! The enhanced YouTube listener provides:")
    print("  ✅ Robust error handling with automatic reconnection")
    print("  ✅ Comprehensive monitoring and status reporting")
    print("  ✅ Flexible configuration system")
    print("  ✅ Advanced timestamp parsing")
    print("  ✅ Thread-safe state management")
    print("  ✅ Exponential backoff retry logic")
    print()
    print("🚀 Ready to handle real-world YouTube stream transcription!")


if __name__ == "__main__":
    main()
