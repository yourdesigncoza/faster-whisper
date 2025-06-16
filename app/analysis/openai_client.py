"""
OpenAI client integration for transcription analysis.

Provides a wrapper around the OpenAI API for analyzing transcribed content.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

from ..utils.config import config


class OpenAIAnalyzer:
    """OpenAI client wrapper for transcription analysis."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI analyzer.
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            model: Model to use (uses config default if not provided)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        self.api_key = api_key or config.openai_api_key
        self.model = model or config.default_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if config.openai_org_id:
            client_kwargs["organization"] = config.openai_org_id
            
        self.client = OpenAI(**client_kwargs)
        
        print(f"✅ OpenAI client initialized with model: {self.model}")
    
    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Make a request to OpenAI API with error handling and retries.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the API call
            
        Returns:
            Response content from OpenAI
        """
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=kwargs.get('max_tokens', config.max_tokens),
                    temperature=kwargs.get('temperature', config.temperature),
                    **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI API request failed after {max_retries} attempts: {e}")
                
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  API request failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
                time.sleep(delay)
    
    def summarize_content(self, content: str, max_length: int = 500) -> str:
        """
        Generate a summary of the transcribed content.
        
        Args:
            content: Transcribed text to summarize
            max_length: Maximum length of summary in words
            
        Returns:
            Summary of the content
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at summarizing transcribed content. "
                    "Create concise, informative summaries that capture the key points and main themes. "
                    f"Keep summaries under {max_length} words."
                )
            },
            {
                "role": "user",
                "content": f"Please summarize this transcribed content:\n\n{content}"
            }
        ]
        
        return self._make_request(messages)
    
    def extract_key_points(self, content: str, num_points: int = 10) -> List[str]:
        """
        Extract key points from transcribed content.
        
        Args:
            content: Transcribed text to analyze
            num_points: Maximum number of key points to extract
            
        Returns:
            List of key points
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at extracting key points from transcribed content. "
                    "Identify the most important insights, decisions, topics, and actionable items. "
                    f"Return up to {num_points} key points as a numbered list."
                )
            },
            {
                "role": "user",
                "content": f"Extract the key points from this transcribed content:\n\n{content}"
            }
        ]
        
        response = self._make_request(messages)
        
        # Parse numbered list into individual points
        points = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets and clean up
                point = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                if point:
                    points.append(point)
        
        return points[:num_points]
    
    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Analyze sentiment of transcribed content.
        
        Args:
            content: Transcribed text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at sentiment analysis. Analyze the overall sentiment, "
                    "emotional tone, and confidence level of transcribed content. "
                    "Provide a structured analysis with sentiment (positive/negative/neutral), "
                    "confidence score (0-1), key emotions detected, and brief explanation."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Analyze the sentiment of this transcribed content and respond in this format:\n"
                    f"Sentiment: [positive/negative/neutral]\n"
                    f"Confidence: [0.0-1.0]\n"
                    f"Key Emotions: [list of emotions]\n"
                    f"Explanation: [brief explanation]\n\n"
                    f"Content:\n{content}"
                )
            }
        ]
        
        response = self._make_request(messages)
        
        # Parse structured response
        result = {
            "sentiment": "neutral",
            "confidence": 0.5,
            "key_emotions": [],
            "explanation": response
        }
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Sentiment:'):
                result["sentiment"] = line.split(':', 1)[1].strip().lower()
            elif line.startswith('Confidence:'):
                try:
                    result["confidence"] = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('Key Emotions:'):
                emotions_str = line.split(':', 1)[1].strip()
                result["key_emotions"] = [e.strip() for e in emotions_str.split(',') if e.strip()]
            elif line.startswith('Explanation:'):
                result["explanation"] = line.split(':', 1)[1].strip()
        
        return result
    
    def custom_analysis(self, content: str, prompt: str) -> str:
        """
        Perform custom analysis with a user-defined prompt.
        
        Args:
            content: Transcribed text to analyze
            prompt: Custom analysis prompt
            
        Returns:
            Analysis result
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert analyst. Provide thorough, insightful analysis based on the user's specific request."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nContent to analyze:\n{content}"
            }
        ]
        
        return self._make_request(messages)
