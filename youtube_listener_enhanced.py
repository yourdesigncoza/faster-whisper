#!/usr/bin/env python3
"""
Enhanced YouTube Stream Listener Script for faster-whisper

This script provides robust YouTube stream transcription with advanced error handling,
automatic reconnection, and comprehensive monitoring capabilities.

Features:
- Robust error handling with automatic recovery
- Exponential backoff for reconnection attempts
- Comprehensive logging and monitoring
- Stream health monitoring
- Graceful degradation and fallback mechanisms
- Configuration-based retry behavior
- Real-time status reporting

Requirements:
- yt-dlp (for YouTube audio extraction)
- faster-whisper (for transcription)
- numpy (for audio processing)
"""

import argparse
import datetime
import logging
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import traceback

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import get_logger


class StreamState(Enum):
    """Stream processing states."""
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


@dataclass
class StreamHealth:
    """Stream health monitoring data."""
    last_audio_time: float = 0.0
    consecutive_failures: int = 0
    total_reconnects: int = 0
    bytes_processed: int = 0
    chunks_processed: int = 0
    transcriptions_completed: int = 0


class EnhancedYouTubeListener:
    """Enhanced YouTube stream listener with robust error handling."""

    def __init__(
        self,
        youtube_url: str,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        output_file: str = "output.txt",
        chunk_duration: float = 30.0,
        language: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the enhanced YouTube listener.

        Args:
            youtube_url: YouTube URL to stream from
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run inference on (cpu, cuda)
            compute_type: Compute type (int8, float16, float32)
            output_file: Output file path for transcriptions
            chunk_duration: Duration of each audio chunk in seconds
            language: Language code (None for auto-detection)
            retry_config: Retry configuration for error handling
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.youtube_url = youtube_url
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.output_file = Path(output_file)
        self.chunk_duration = chunk_duration
        self.language = language
        self.retry_config = retry_config or RetryConfig()

        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # State management
        self.state = StreamState.INITIALIZING
        self.is_running = False
        self.should_stop = False
        self.state_lock = threading.Lock()
        
        # Stream health monitoring
        self.health = StreamHealth()
        self.health_lock = threading.Lock()
        
        # Audio processing
        self.audio_process = None
        self.audio_thread = None
        self.current_audio_url = None
        self.start_timestamp = None
        
        # Whisper model
        self.model = None
        self.transcription_thread = None
        self.transcription_queue = []
        self.queue_lock = threading.Lock()

        # Monitoring thread
        self.monitor_thread = None
        self.last_activity_time = time.time()

        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup enhanced logging with proper formatting."""
        logger = get_logger()
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _signal_handler(self, _signum, _frame):
        """Handle Ctrl+C gracefully."""
        self.logger.info("🛑 Received shutdown signal, stopping gracefully...")
        self.stop()
        sys.exit(0)

    def _update_state(self, new_state: StreamState):
        """Thread-safe state update."""
        with self.state_lock:
            old_state = self.state
            self.state = new_state
            self.logger.info(f"State transition: {old_state.value} → {new_state.value}")

    def _update_health(self, **kwargs):
        """Thread-safe health update."""
        with self.health_lock:
            for key, value in kwargs.items():
                if hasattr(self.health, key):
                    setattr(self.health, key, value)
            self.last_activity_time = time.time()

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = min(
            self.retry_config.initial_delay * (self.retry_config.backoff_multiplier ** attempt),
            self.retry_config.max_delay
        )
        
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay

    def _retry_with_backoff(self, operation, operation_name: str, *args, **kwargs):
        """Execute operation with retry logic and exponential backoff."""
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"✅ {operation_name} succeeded after {attempt} retries")
                return result
            except Exception as e:
                self._update_health(consecutive_failures=self.health.consecutive_failures + 1)
                
                if attempt == self.retry_config.max_retries:
                    self.logger.error(f"❌ {operation_name} failed after {attempt + 1} attempts: {e}")
                    raise
                
                delay = self._calculate_retry_delay(attempt)
                self.logger.warning(
                    f"⚠️  {operation_name} failed (attempt {attempt + 1}/{self.retry_config.max_retries + 1}): {e}"
                )
                self.logger.info(f"🔄 Retrying in {delay:.1f}s...")
                
                if self.should_stop:
                    raise KeyboardInterrupt("Stop requested during retry")
                
                time.sleep(delay)

    def _parse_timestamp_from_url(self, url: str) -> Optional[int]:
        """Parse timestamp from YouTube URL and return seconds."""
        try:
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)
            timestamp_param = None

            if 't' in query_params:
                timestamp_param = query_params['t'][0]

            if not timestamp_param and parsed.fragment:
                if parsed.fragment.startswith('t='):
                    timestamp_param = parsed.fragment[2:]
                elif parsed.fragment.isdigit():
                    timestamp_param = parsed.fragment

            if not timestamp_param:
                return None

            return self._parse_timestamp_string(timestamp_param)

        except Exception as e:
            self.logger.warning(f"Could not parse timestamp from URL: {e}")
            return None

    def _parse_timestamp_string(self, timestamp_str: str) -> int:
        """Parse timestamp string into seconds."""
        timestamp_str = timestamp_str.strip().lower()

        if re.match(r'^\d+s?$', timestamp_str):
            return int(timestamp_str.rstrip('s'))

        total_seconds = 0
        hours_match = re.search(r'(\d+)h', timestamp_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600

        minutes_match = re.search(r'(\d+)m', timestamp_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60

        seconds_match = re.search(r'(\d+)s', timestamp_str)
        if seconds_match:
            total_seconds += int(seconds_match.group(1))

        if total_seconds == 0:
            try:
                total_seconds = int(float(timestamp_str))
            except ValueError:
                raise ValueError(f"Invalid timestamp format: {timestamp_str}")

        return total_seconds

    def _load_model(self):
        """Load the Whisper model with retry logic."""
        def load_operation():
            self.logger.info(f"🤖 Loading Whisper model '{self.model_size}' on {self.device}...")
            model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self.logger.info("✅ Model loaded successfully!")
            return model

        self.model = self._retry_with_backoff(load_operation, "Model loading")

    def _validate_youtube_url(self):
        """Validate the YouTube URL."""
        try:
            parsed = urllib.parse.urlparse(self.youtube_url)
            if not (parsed.netloc in ['www.youtube.com', 'youtube.com', 'youtu.be', 'm.youtube.com']):
                raise ValueError("Invalid YouTube URL")
            self.logger.info(f"✅ Valid YouTube URL: {self.youtube_url}")
        except Exception as e:
            self.logger.error(f"❌ Invalid YouTube URL: {e}")
            raise

    def _setup_youtube_stream(self) -> Tuple[str, Optional[int]]:
        """Setup YouTube audio stream using yt-dlp with retry logic."""
        def stream_setup_operation():
            self.start_timestamp = self._parse_timestamp_from_url(self.youtube_url)

            if self.start_timestamp is not None:
                if not self._validate_timestamp(self.start_timestamp):
                    self.logger.warning("Invalid timestamp detected, ignoring seeking")
                    self.start_timestamp = None
                else:
                    duration = self._get_video_duration(self.youtube_url)
                    if duration is not None and self.start_timestamp >= duration:
                        self.logger.warning(
                            f"Timestamp {self.start_timestamp}s is beyond video duration {duration}s, "
                            "starting from beginning"
                        )
                        self.start_timestamp = None

            cmd = [
                'yt-dlp',
                '--format', 'bestaudio/best',
                '--get-url',
                '--no-playlist',
            ]

            if self.start_timestamp is not None:
                section = f"*{self.start_timestamp}-inf"
                cmd.extend(['--download-sections', section])
                self.logger.info(f"🕐 Seeking to timestamp: {self.start_timestamp}s")

            cmd.append(self.youtube_url)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            audio_url = result.stdout.strip()

            if not audio_url:
                raise ValueError("Could not extract audio URL from YouTube")

            self.logger.info("🎵 Audio stream URL extracted successfully")
            return audio_url, self.start_timestamp

        return self._retry_with_backoff(stream_setup_operation, "YouTube stream setup")

    def _validate_timestamp(self, timestamp: int) -> bool:
        """Validate timestamp is reasonable."""
        if timestamp < 0:
            self.logger.warning(f"Negative timestamp {timestamp}s is invalid")
            return False

        if timestamp > 86400:  # 24 hours
            self.logger.warning(f"Timestamp {timestamp}s seems too large")
            return False

        return True

    def _get_video_duration(self, url: str) -> Optional[int]:
        """Get video duration using yt-dlp."""
        try:
            cmd = ['yt-dlp', '--get-duration', '--no-playlist', url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
            duration_str = result.stdout.strip()

            if not duration_str:
                return None

            parts = duration_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            else:
                return int(parts[0])  # Just seconds

        except Exception as e:
            self.logger.warning(f"Could not get video duration: {e}")
            return None

    def _has_speech_content(self, audio_data):
        """Quick check to see if audio chunk likely contains speech."""
        rms = np.sqrt(np.mean(audio_data**2))
        max_amplitude = np.max(np.abs(audio_data))

        if rms < 0.001 or max_amplitude < 0.01:
            return False, "too quiet"

        if max_amplitude > 1:
            return False, "too loud (likely music)"

        zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
        zcr = zero_crossings / len(audio_data)

        if zcr < 0.005:
            return False, "too monotone (likely music/tone)"
        if zcr > 0.4:
            return False, "too noisy"

        return True, f"rms={rms:.4f}, zcr={zcr:.4f}"

    def _create_audio_process(self, audio_url: str) -> subprocess.Popen:
        """Create FFmpeg audio processing subprocess."""
        cmd = ['ffmpeg', '-loglevel', 'error']

        if self.start_timestamp is not None:
            cmd.extend(['-ss', str(self.start_timestamp)])

        cmd.extend([
            '-i', audio_url,
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-'
        ])

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )

    def _audio_processing_worker(self):
        """Enhanced audio processing worker with automatic reconnection."""
        consecutive_failures = 0

        while self.is_running and not self.should_stop:
            try:
                self._update_state(StreamState.CONNECTING)

                # Setup stream with retry logic
                audio_url, start_timestamp = self._retry_with_backoff(
                    self._setup_youtube_stream, "Audio stream setup"
                )
                self.current_audio_url = audio_url

                # Create audio process
                self.audio_process = self._create_audio_process(audio_url)
                self.logger.info("🎵 Starting audio stream processing...")

                self._update_state(StreamState.STREAMING)
                consecutive_failures = 0  # Reset on successful connection

                # Process audio in chunks
                chunk_size = int(16000 * self.chunk_duration)
                audio_buffer = b''
                last_data_time = time.time()

                while self.is_running and not self.should_stop:
                    try:
                        # Read audio data with timeout
                        data = self.audio_process.stdout.read(chunk_size * 2)

                        if not data:
                            # Check if process is still alive
                            if self.audio_process.poll() is not None:
                                stderr_output = self.audio_process.stderr.read().decode('utf-8', errors='ignore')
                                if stderr_output:
                                    self.logger.warning(f"FFmpeg stderr: {stderr_output}")
                                raise ConnectionError("Audio stream ended unexpectedly")

                            # Check for timeout
                            if time.time() - last_data_time > 30:  # 30 second timeout
                                raise TimeoutError("No audio data received for 30 seconds")

                            time.sleep(0.1)
                            continue

                        last_data_time = time.time()
                        audio_buffer += data
                        self._update_health(
                            last_audio_time=time.time(),
                            bytes_processed=self.health.bytes_processed + len(data)
                        )

                        # Process when we have enough data
                        if len(audio_buffer) >= chunk_size * 2:
                            audio_data = np.frombuffer(audio_buffer[:chunk_size * 2], dtype=np.int16)
                            audio_float = audio_data.astype(np.float32) / 32768.0

                            # Add to transcription queue
                            with self.queue_lock:
                                self.transcription_queue.append(audio_float)

                            self.logger.debug(f"📝 Queued {self.chunk_duration:.1f}s of audio for transcription")
                            self._update_health(chunks_processed=self.health.chunks_processed + 1)

                            # Remove processed data from buffer
                            audio_buffer = audio_buffer[chunk_size * 2:]

                    except (ConnectionError, TimeoutError, OSError) as e:
                        self.logger.warning(f"Audio processing error: {e}")
                        break  # Break inner loop to reconnect

            except Exception as e:
                consecutive_failures += 1
                self._update_health(consecutive_failures=consecutive_failures)

                if consecutive_failures >= self.retry_config.max_retries:
                    self.logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), stopping")
                    self._update_state(StreamState.ERROR)
                    break

                self._update_state(StreamState.RECONNECTING)
                delay = self._calculate_retry_delay(consecutive_failures - 1)
                self.logger.warning(
                    f"⚠️  Audio processing failed (attempt {consecutive_failures}): {e}"
                )
                self.logger.info(f"🔄 Reconnecting in {delay:.1f}s...")

                # Clean up current process
                if self.audio_process:
                    try:
                        self.audio_process.terminate()
                        self.audio_process.wait(timeout=5)
                    except:
                        pass
                    self.audio_process = None

                if not self.should_stop:
                    time.sleep(delay)
                    self._update_health(total_reconnects=self.health.total_reconnects + 1)

        self._update_state(StreamState.STOPPED)
        self.logger.info("🛑 Audio processing worker stopped")

    def _transcription_worker(self):
        """Enhanced transcription worker with error handling."""
        while self.is_running and not self.should_stop:
            audio_data = None

            # Get audio from queue
            with self.queue_lock:
                if self.transcription_queue:
                    audio_data = self.transcription_queue.pop(0)

            if audio_data is not None:
                try:
                    # Quick pre-check for speech content
                    has_speech, reason = self._has_speech_content(audio_data)
                    self.logger.debug(f"📊 Audio analysis: {reason}")

                    if not has_speech:
                        self.logger.debug(f"⏭️  Skipping transcription: {reason}")
                        continue

                    # Transcribe audio
                    self.logger.debug("🔄 Transcribing audio...")

                    segments, _ = self.model.transcribe(
                        audio_data,
                        language=self.language,
                        vad_filter=True,
                        word_timestamps=False
                    )

                    # Collect transcription text
                    transcription = ""
                    segment_count = 0
                    for segment in segments:
                        transcription += segment.text
                        segment_count += 1

                    self.logger.debug(f"📈 Found {segment_count} segments")

                    # Filter out meaningless transcriptions
                    cleaned_text = transcription.strip()

                    if not cleaned_text or len(cleaned_text.replace(".", "").replace(" ", "")) < 2:
                        self.logger.debug("🔇 No meaningful speech detected")
                        continue

                    # Skip common noise patterns
                    noise_patterns = ["...", ". . .", ".", " ", "uh", "um", "ah"]
                    if cleaned_text.lower() in noise_patterns:
                        self.logger.debug(f"⏭️  Skipping noise pattern: '{cleaned_text}'")
                        continue

                    # Save meaningful transcriptions
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    output_line = f"[{timestamp}] {cleaned_text}\n"

                    # Append to file with error handling
                    try:
                        with open(self.output_file, "a", encoding="utf-8") as f:
                            f.write(output_line)

                        self.logger.info(f"✅ Transcribed: {cleaned_text}")
                        self._update_health(transcriptions_completed=self.health.transcriptions_completed + 1)

                    except IOError as e:
                        self.logger.error(f"❌ Failed to write transcription: {e}")

                except Exception as e:
                    self.logger.error(f"❌ Transcription error: {e}")
                    self.logger.debug(f"Transcription error details: {traceback.format_exc()}")
            else:
                # No audio to process, sleep briefly
                time.sleep(0.1)

        self.logger.info("🛑 Transcription worker stopped")

    def _monitoring_worker(self):
        """Monitor stream health and provide status updates."""
        last_report_time = time.time()
        report_interval = 60  # Report every 60 seconds

        while self.is_running and not self.should_stop:
            try:
                current_time = time.time()

                # Check for stalled processing
                if current_time - self.last_activity_time > 120:  # 2 minutes without activity
                    self.logger.warning("⚠️  No activity detected for 2 minutes")

                # Periodic status report
                if current_time - last_report_time >= report_interval:
                    self._report_status()
                    last_report_time = current_time

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(30)  # Wait longer on error

        self.logger.info("🛑 Monitoring worker stopped")

    def _report_status(self):
        """Report current stream status and health metrics."""
        with self.health_lock:
            uptime = time.time() - self.last_activity_time

            self.logger.info(
                f"📊 Status Report - State: {self.state.value} | "
                f"Chunks: {self.health.chunks_processed} | "
                f"Transcriptions: {self.health.transcriptions_completed} | "
                f"Reconnects: {self.health.total_reconnects} | "
                f"Failures: {self.health.consecutive_failures}"
            )

    def start(self):
        """Start the enhanced YouTube listener."""
        try:
            self.logger.info("🚀 Starting Enhanced YouTube Listener...")
            self.logger.info(f"🎵 YouTube URL: {self.youtube_url}")
            self.logger.info(f"📁 Output file: {self.output_file.absolute()}")
            self.logger.info(f"🔧 Retry config: max_retries={self.retry_config.max_retries}, "
                           f"max_delay={self.retry_config.max_delay}s")
            self.logger.info("💡 Enhanced listener with auto-reconnection active. Press Ctrl+C to stop.")
            self.logger.info("-" * 70)

            # Validate YouTube URL
            self._validate_youtube_url()

            # Load model
            self._load_model()

            # Create output file if it doesn't exist
            self.output_file.touch()

            # Start processing
            self.is_running = True
            self.should_stop = False

            # Start worker threads
            self.transcription_thread = threading.Thread(target=self._transcription_worker, daemon=True)
            self.transcription_thread.start()

            self.audio_thread = threading.Thread(target=self._audio_processing_worker, daemon=True)
            self.audio_thread.start()

            self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitor_thread.start()

            # Keep main thread alive and monitor for stop conditions
            try:
                while self.is_running and not self.should_stop:
                    # Check if critical threads are still alive
                    if not self.audio_thread.is_alive():
                        self.logger.error("❌ Audio thread died unexpectedly")
                        break

                    if not self.transcription_thread.is_alive():
                        self.logger.error("❌ Transcription thread died unexpectedly")
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                self.logger.info("🛑 Keyboard interrupt received")

        except Exception as e:
            self.logger.error(f"❌ Failed to start listener: {e}")
            self.logger.debug(f"Startup error details: {traceback.format_exc()}")
            raise
        finally:
            self.stop()

    def stop(self):
        """Stop the enhanced YouTube listener."""
        self.logger.info("🛑 Stopping Enhanced YouTube Listener...")

        self.should_stop = True
        self.is_running = False
        self._update_state(StreamState.STOPPING)

        # Stop audio process
        if self.audio_process:
            try:
                self.audio_process.terminate()
                self.audio_process.wait(timeout=5)
            except:
                try:
                    self.audio_process.kill()
                except:
                    pass

        # Wait for threads to finish
        threads_to_join = [
            ("audio processing", self.audio_thread),
            ("transcription", self.transcription_thread),
            ("monitoring", self.monitor_thread)
        ]

        for thread_name, thread in threads_to_join:
            if thread and thread.is_alive():
                self.logger.info(f"⏳ Stopping {thread_name}...")
                thread.join(timeout=10)
                if thread.is_alive():
                    self.logger.warning(f"⚠️  {thread_name} thread did not stop gracefully")

        self._update_state(StreamState.STOPPED)
        self._report_status()
        self.logger.info("✅ Enhanced YouTube Listener stopped successfully!")

    def get_status(self) -> Dict[str, Any]:
        """Get current status and health information."""
        with self.health_lock, self.state_lock:
            return {
                "state": self.state.value,
                "is_running": self.is_running,
                "health": {
                    "last_audio_time": self.health.last_audio_time,
                    "consecutive_failures": self.health.consecutive_failures,
                    "total_reconnects": self.health.total_reconnects,
                    "bytes_processed": self.health.bytes_processed,
                    "chunks_processed": self.health.chunks_processed,
                    "transcriptions_completed": self.health.transcriptions_completed,
                },
                "config": {
                    "youtube_url": self.youtube_url,
                    "model_size": self.model_size,
                    "device": self.device,
                    "output_file": str(self.output_file),
                    "retry_config": {
                        "max_retries": self.retry_config.max_retries,
                        "max_delay": self.retry_config.max_delay,
                    }
                }
            }


def main():
    """Main entry point with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced YouTube stream transcription with robust error handling",
        epilog="""
Examples:
  %(prog)s --url "https://youtu.be/jGpWRfxHFcs?t=863"
  %(prog)s --url "https://www.youtube.com/watch?v=jGpWRfxHFcs&t=14m23s" --model small
  %(prog)s --url "https://youtu.be/jGpWRfxHFcs" --max-retries 10 --log-level DEBUG
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic options
    parser.add_argument("--url", required=False,
                       help="YouTube URL to transcribe (supports timestamp parameters)")
    parser.add_argument("--model", default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Whisper model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on")
    parser.add_argument("--compute-type", default="int8",
                       choices=["int8", "float16", "float32"],
                       help="Compute type for inference")
    parser.add_argument("--output", default="output.txt",
                       help="Output file path")
    parser.add_argument("--language", default=None,
                       help="Language code (e.g., 'en', 'es', 'fr')")
    parser.add_argument("--chunk-duration", type=float, default=30.0,
                       help="Duration of each audio chunk in seconds")

    # Enhanced error handling options
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum number of retry attempts")
    parser.add_argument("--initial-delay", type=float, default=1.0,
                       help="Initial retry delay in seconds")
    parser.add_argument("--max-delay", type=float, default=60.0,
                       help="Maximum retry delay in seconds")
    parser.add_argument("--backoff-multiplier", type=float, default=2.0,
                       help="Backoff multiplier for retry delays")
    parser.add_argument("--no-jitter", action="store_true",
                       help="Disable jitter in retry delays")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Get YouTube URL from user if not provided
    youtube_url = args.url
    if not youtube_url:
        print("🎵 Enhanced YouTube Stream Transcriber")
        print("Features: Auto-reconnection, robust error handling, comprehensive monitoring")
        print()
        print("Supports timestamped URLs like:")
        print("  - https://youtu.be/jGpWRfxHFcs?t=863")
        print("  - https://www.youtube.com/watch?v=jGpWRfxHFcs&t=14m23s")
        print("  - https://youtu.be/jGpWRfxHFcs#t=1h14m23s")
        print()
        print("Please enter a YouTube URL:")
        youtube_url = input("URL: ").strip()

        if not youtube_url:
            print("❌ No URL provided. Exiting.")
            sys.exit(1)

    # Create retry configuration
    retry_config = RetryConfig(
        max_retries=args.max_retries,
        initial_delay=args.initial_delay,
        max_delay=args.max_delay,
        backoff_multiplier=args.backoff_multiplier,
        jitter=not args.no_jitter
    )

    # Create enhanced listener
    listener = EnhancedYouTubeListener(
        youtube_url=youtube_url,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        output_file=args.output,
        language=args.language,
        chunk_duration=args.chunk_duration,
        retry_config=retry_config,
        log_level=args.log_level,
    )

    # Start listening
    try:
        listener.start()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
