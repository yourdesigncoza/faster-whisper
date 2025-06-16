#!/usr/bin/env python3
"""
Test script for YouTube timestamp seeking functionality.

This script tests the timestamp parsing and seeking capabilities
of the YouTube listener without actually downloading or processing audio.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import youtube_listener
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from youtube_listener import YouTubeListener


class TestTimestampSeeking(unittest.TestCase):
    """Test cases for timestamp seeking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.listener = YouTubeListener(
            youtube_url="https://www.youtube.com/watch?v=test",
            chunk_duration=10.0
        )

    def test_parse_timestamp_simple_seconds(self):
        """Test parsing simple seconds format."""
        test_cases = [
            ("https://youtu.be/jGpWRfxHFcs?t=863", 863),
            ("https://youtu.be/jGpWRfxHFcs?t=863s", 863),
            ("https://www.youtube.com/watch?v=jGpWRfxHFcs&t=863", 863),
            ("https://www.youtube.com/watch?v=jGpWRfxHFcs&t=863s", 863),
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.listener._parse_timestamp_from_url(url)
                self.assertEqual(result, expected)

    def test_parse_timestamp_fragment(self):
        """Test parsing timestamp from URL fragment."""
        test_cases = [
            ("https://www.youtube.com/watch?v=jGpWRfxHFcs#t=863", 863),
            ("https://youtu.be/jGpWRfxHFcs#t=863s", 863),
            ("https://youtu.be/jGpWRfxHFcs#863", 863),
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.listener._parse_timestamp_from_url(url)
                self.assertEqual(result, expected)

    def test_parse_timestamp_complex_format(self):
        """Test parsing complex timestamp formats."""
        test_cases = [
            ("https://youtu.be/jGpWRfxHFcs?t=14m23s", 14*60 + 23),  # 863 seconds
            ("https://youtu.be/jGpWRfxHFcs?t=1h14m23s", 1*3600 + 14*60 + 23),  # 4463 seconds
            ("https://youtu.be/jGpWRfxHFcs?t=2h30m", 2*3600 + 30*60),  # 9000 seconds
            ("https://youtu.be/jGpWRfxHFcs?t=45m", 45*60),  # 2700 seconds
            ("https://youtu.be/jGpWRfxHFcs?t=3h", 3*3600),  # 10800 seconds
        ]
        
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.listener._parse_timestamp_from_url(url)
                self.assertEqual(result, expected)

    def test_parse_timestamp_no_timestamp(self):
        """Test URLs without timestamps."""
        test_cases = [
            "https://www.youtube.com/watch?v=jGpWRfxHFcs",
            "https://youtu.be/jGpWRfxHFcs",
            "https://www.youtube.com/watch?v=jGpWRfxHFcs&list=PLtest",
        ]
        
        for url in test_cases:
            with self.subTest(url=url):
                result = self.listener._parse_timestamp_from_url(url)
                self.assertIsNone(result)

    def test_parse_timestamp_string(self):
        """Test timestamp string parsing."""
        test_cases = [
            ("863", 863),
            ("863s", 863),
            ("14m23s", 14*60 + 23),
            ("1h14m23s", 1*3600 + 14*60 + 23),
            ("2h30m", 2*3600 + 30*60),
            ("45m", 45*60),
            ("3h", 3*3600),
        ]
        
        for timestamp_str, expected in test_cases:
            with self.subTest(timestamp_str=timestamp_str):
                result = self.listener._parse_timestamp_string(timestamp_str)
                self.assertEqual(result, expected)

    def test_parse_timestamp_string_invalid(self):
        """Test invalid timestamp string formats."""
        invalid_cases = [
            "invalid",
            "1x2y3z",
            "",
            "abc123",
        ]
        
        for timestamp_str in invalid_cases:
            with self.subTest(timestamp_str=timestamp_str):
                with self.assertRaises(ValueError):
                    self.listener._parse_timestamp_string(timestamp_str)

    def test_validate_timestamp(self):
        """Test timestamp validation."""
        valid_cases = [0, 1, 863, 3600, 7200, 86399]  # 0 to just under 24 hours
        invalid_cases = [-1, -100, 86401, 100000]  # negative or over 24 hours
        
        for timestamp in valid_cases:
            with self.subTest(timestamp=timestamp):
                result = self.listener._validate_timestamp(timestamp)
                self.assertTrue(result)
        
        for timestamp in invalid_cases:
            with self.subTest(timestamp=timestamp):
                result = self.listener._validate_timestamp(timestamp)
                self.assertFalse(result)

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        test_cases = [
            (0, "00:00"),
            (30, "00:30"),
            (90, "01:30"),
            (3661, "01:01:01"),
            (7323, "02:02:03"),
        ]
        
        for seconds, expected in test_cases:
            with self.subTest(seconds=seconds):
                result = self.listener._format_timestamp(seconds)
                self.assertEqual(result, expected)

    @patch('subprocess.run')
    def test_get_video_duration(self, mock_run):
        """Test video duration retrieval."""
        # Mock successful duration retrieval
        mock_run.return_value.stdout = "14:23\n"
        mock_run.return_value.returncode = 0
        
        result = self.listener._get_video_duration("https://youtu.be/test")
        expected = 14*60 + 23  # 863 seconds
        self.assertEqual(result, expected)
        
        # Test with hours
        mock_run.return_value.stdout = "1:14:23\n"
        result = self.listener._get_video_duration("https://youtu.be/test")
        expected = 1*3600 + 14*60 + 23  # 4463 seconds
        self.assertEqual(result, expected)

    @patch('subprocess.run')
    def test_setup_youtube_stream_with_timestamp(self, mock_run):
        """Test YouTube stream setup with timestamp."""
        # Mock yt-dlp command success
        mock_run.return_value.stdout = "https://example.com/audio.m4a\n"
        mock_run.return_value.returncode = 0
        
        # Test URL with timestamp
        listener = YouTubeListener(
            youtube_url="https://youtu.be/jGpWRfxHFcs?t=863",
            chunk_duration=10.0
        )
        
        audio_url, timestamp = listener._setup_youtube_stream()
        
        self.assertEqual(audio_url, "https://example.com/audio.m4a")
        self.assertEqual(timestamp, 863)
        
        # Verify yt-dlp was called with download-sections
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]  # Get the command arguments
        self.assertIn('--download-sections', call_args)
        self.assertIn('*863-inf', call_args)


def run_interactive_test():
    """Run an interactive test to verify timestamp parsing."""
    print("🧪 Interactive Timestamp Parsing Test")
    print("=" * 50)
    
    test_urls = [
        "https://youtu.be/jGpWRfxHFcs?t=863",
        "https://youtu.be/jGpWRfxHFcs?t=863s", 
        "https://www.youtube.com/watch?v=jGpWRfxHFcs&t=14m23s",
        "https://www.youtube.com/watch?v=jGpWRfxHFcs#t=1h14m23s",
        "https://www.youtube.com/watch?v=jGpWRfxHFcs",  # No timestamp
    ]
    
    listener = YouTubeListener(
        youtube_url="https://www.youtube.com/watch?v=test",
        chunk_duration=10.0
    )
    
    for url in test_urls:
        timestamp = listener._parse_timestamp_from_url(url)
        if timestamp is not None:
            formatted = listener._format_timestamp(timestamp)
            print(f"✅ {url}")
            print(f"   → {timestamp}s ({formatted})")
        else:
            print(f"❌ {url}")
            print(f"   → No timestamp found")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test timestamp seeking functionality")
    parser.add_argument("--interactive", action="store_true", 
                       help="Run interactive test")
    parser.add_argument("--unittest", action="store_true",
                       help="Run unit tests")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_test()
    elif args.unittest:
        unittest.main(argv=[''])
    else:
        print("Usage: python test_timestamp_seeking.py [--interactive|--unittest]")
        print("  --interactive: Run interactive timestamp parsing test")
        print("  --unittest: Run unit tests")
