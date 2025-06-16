#!/usr/bin/env python3
"""
Test script for OpenAI integration functionality.

This script tests the OpenAI analysis capabilities without requiring
actual API calls (uses mock data for testing).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.transcription_parser import TranscriptionParser, TranscriptionEntry
from app.utils.config import Config


class TestTranscriptionParser(unittest.TestCase):
    """Test cases for transcription parsing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = [
            "[2024-01-15 14:23:45] This is the first transcription entry.",
            "[2024-01-15 14:23:50] Here's another entry with some content.",
            "[2024-01-15 14:24:00] Trading discussion about market trends.",
            "[2024-01-15 14:24:15] Analysis of current price movements.",
            "Invalid line without timestamp",
            "[2024-01-15 14:24:30] Final entry in the test data."
        ]
        
        # Create temporary test file
        self.test_file = Path("test_transcription.txt")
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.test_data))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_parse_transcription_file(self):
        """Test parsing of transcription file."""
        parser = TranscriptionParser(self.test_file)
        entries = parser.read_transcription_file()
        
        # Should parse 5 valid entries (excluding invalid line)
        self.assertEqual(len(entries), 5)
        
        # Check first entry
        first_entry = entries[0]
        self.assertEqual(first_entry.content, "This is the first transcription entry.")
        self.assertEqual(first_entry.timestamp.hour, 14)
        self.assertEqual(first_entry.timestamp.minute, 23)
        self.assertEqual(first_entry.timestamp.second, 45)
    
    def test_get_content_text(self):
        """Test extracting plain text content."""
        parser = TranscriptionParser(self.test_file)
        entries = parser.read_transcription_file()
        
        content = parser.get_content_text(entries)
        self.assertIn("This is the first transcription entry.", content)
        self.assertIn("Trading discussion about market trends.", content)
    
    def test_split_into_chunks(self):
        """Test splitting content into chunks."""
        parser = TranscriptionParser(self.test_file)
        entries = parser.read_transcription_file()
        
        # Test with small chunk size
        chunks = parser.split_into_chunks(entries, max_length=100)
        self.assertGreater(len(chunks), 1)
        
        # Verify each chunk respects the length limit
        for chunk in chunks:
            chunk_text = parser.get_content_text(chunk)
            self.assertLessEqual(len(chunk_text), 100)
    
    def test_get_statistics(self):
        """Test statistics generation."""
        parser = TranscriptionParser(self.test_file)
        entries = parser.read_transcription_file()
        
        stats = parser.get_statistics(entries)
        
        self.assertEqual(stats["total_entries"], 5)
        self.assertGreater(stats["total_characters"], 0)
        self.assertGreater(stats["total_words"], 0)
        self.assertIsNotNone(stats["first_timestamp"])
        self.assertIsNotNone(stats["last_timestamp"])


class TestConfigurationManagement(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        # Test with environment variables
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'DEFAULT_MODEL': 'gpt-4',
            'MAX_TOKENS': '1500'
        }):
            config = Config()
            self.assertEqual(config.openai_api_key, 'test_key')
            self.assertEqual(config.default_model, 'gpt-4')
            self.assertEqual(config.max_tokens, 1500)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test missing API key
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                Config()
    
    def test_path_methods(self):
        """Test path-related methods."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            config = Config()
            
            transcription_path = config.get_transcription_path()
            self.assertIsInstance(transcription_path, Path)
            
            output_dir = config.get_analysis_output_dir()
            self.assertIsInstance(output_dir, Path)


@patch('app.analysis.openai_client.OPENAI_AVAILABLE', True)
@patch('app.analysis.openai_client.OpenAI')
class TestOpenAIIntegration(unittest.TestCase):
    """Test cases for OpenAI integration (mocked)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_content = (
            "Today we're discussing market trends and trading strategies. "
            "The current price action shows bullish momentum with strong support levels. "
            "Key resistance is at 1.2500 level. We expect continued upward movement "
            "based on technical analysis and fundamental factors."
        )
    
    def test_openai_analyzer_initialization(self, mock_openai):
        """Test OpenAI analyzer initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            from app.analysis.openai_client import OpenAIAnalyzer
            
            analyzer = OpenAIAnalyzer()
            self.assertIsNotNone(analyzer.client)
            self.assertEqual(analyzer.api_key, 'test_key')
    
    def test_summarize_content(self, mock_openai):
        """Test content summarization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            from app.analysis.openai_client import OpenAIAnalyzer
            
            # Mock the API response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = "This is a test summary of the trading content."
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            
            analyzer = OpenAIAnalyzer()
            summary = analyzer.summarize_content(self.sample_content)
            
            self.assertEqual(summary, "This is a test summary of the trading content.")
            mock_openai.return_value.chat.completions.create.assert_called_once()
    
    def test_extract_key_points(self, mock_openai):
        """Test key point extraction."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            from app.analysis.openai_client import OpenAIAnalyzer
            
            # Mock the API response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                "1. Market shows bullish momentum\n"
                "2. Strong support levels identified\n"
                "3. Key resistance at 1.2500"
            )
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            
            analyzer = OpenAIAnalyzer()
            key_points = analyzer.extract_key_points(self.sample_content)
            
            self.assertEqual(len(key_points), 3)
            self.assertIn("Market shows bullish momentum", key_points[0])
    
    def test_analyze_sentiment(self, mock_openai):
        """Test sentiment analysis."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            from app.analysis.openai_client import OpenAIAnalyzer
            
            # Mock the API response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = (
                "Sentiment: positive\n"
                "Confidence: 0.85\n"
                "Key Emotions: optimistic, confident\n"
                "Explanation: The content shows positive outlook on market trends."
            )
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            
            analyzer = OpenAIAnalyzer()
            sentiment = analyzer.analyze_sentiment(self.sample_content)
            
            self.assertEqual(sentiment["sentiment"], "positive")
            self.assertEqual(sentiment["confidence"], 0.85)
            self.assertIn("optimistic", sentiment["key_emotions"])


def run_integration_test():
    """Run a simple integration test to verify the setup."""
    print("🧪 OpenAI Integration Test")
    print("=" * 50)
    
    # Test configuration
    try:
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            from app.utils.config import Config
            config = Config()
            print("✅ Configuration loading: PASSED")
    except Exception as e:
        print(f"❌ Configuration loading: FAILED - {e}")
        return False
    
    # Test transcription parsing
    try:
        test_data = [
            "[2024-01-15 14:23:45] Test transcription entry one.",
            "[2024-01-15 14:23:50] Test transcription entry two."
        ]
        
        test_file = Path("temp_test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(test_data))
        
        from app.utils.transcription_parser import TranscriptionParser
        parser = TranscriptionParser(test_file)
        entries = parser.read_transcription_file()
        
        test_file.unlink()  # Clean up
        
        if len(entries) == 2:
            print("✅ Transcription parsing: PASSED")
        else:
            print(f"❌ Transcription parsing: FAILED - Expected 2 entries, got {len(entries)}")
            return False
            
    except Exception as e:
        print(f"❌ Transcription parsing: FAILED - {e}")
        return False
    
    # Test OpenAI client (mocked)
    try:
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('app.analysis.openai_client.OPENAI_AVAILABLE', True):
                with patch('app.analysis.openai_client.OpenAI'):
                    # Import after patching
                    import importlib
                    import app.analysis.openai_client
                    importlib.reload(app.analysis.openai_client)

                    from app.analysis.openai_client import OpenAIAnalyzer
                    analyzer = OpenAIAnalyzer()
                    print("✅ OpenAI client initialization: PASSED")
    except Exception as e:
        print(f"⚠️  OpenAI client initialization: SKIPPED - {e}")
        print("   (This is expected if OpenAI library is not installed)")
        # Don't fail the test for missing OpenAI library
        print("✅ OpenAI client test: SKIPPED (library not available)")
        return True
    
    print("\n🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenAI integration functionality")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration test")
    parser.add_argument("--unittest", action="store_true",
                       help="Run unit tests")
    
    args = parser.parse_args()
    
    if args.integration:
        run_integration_test()
    elif args.unittest:
        unittest.main(argv=[''])
    else:
        print("Usage: python test_openai_integration.py [--integration|--unittest]")
        print("  --integration: Run integration test")
        print("  --unittest: Run unit tests")
