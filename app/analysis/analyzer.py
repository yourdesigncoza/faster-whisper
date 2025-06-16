"""
Main analysis module for transcription processing.

Combines transcription parsing with OpenAI analysis capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .openai_client import OpenAIAnalyzer
from ..utils.transcription_parser import TranscriptionParser, TranscriptionEntry
from ..utils.config import config


class TranscriptionAnalyzer:
    """Main analyzer class that combines parsing and AI analysis."""
    
    def __init__(self, transcription_file: Optional[Path] = None):
        """
        Initialize the transcription analyzer.
        
        Args:
            transcription_file: Path to transcription file (uses config default if not provided)
        """
        self.parser = TranscriptionParser(transcription_file)
        self.openai_analyzer = OpenAIAnalyzer()
        self.entries: List[TranscriptionEntry] = []
        
    def load_transcription(self) -> List[TranscriptionEntry]:
        """
        Load transcription data from file.
        
        Returns:
            List of transcription entries
        """
        self.entries = self.parser.read_transcription_file()
        return self.entries
    
    def analyze_full_transcription(self, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on the full transcription.
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.entries:
            self.load_transcription()
        
        if not self.entries:
            raise ValueError("No transcription data found")
        
        print("🔍 Starting comprehensive transcription analysis...")
        
        # Get full content
        full_content = self.parser.get_content_text(self.entries)
        
        # Perform various analyses
        results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "transcription_file": str(self.parser.file_path),
                "statistics": self.parser.get_statistics(self.entries)
            },
            "summary": self._analyze_with_chunking(full_content, "summary"),
            "key_points": self._analyze_with_chunking(full_content, "key_points"),
            "sentiment": self._analyze_with_chunking(full_content, "sentiment"),
        }
        
        if save_results:
            self._save_analysis_results(results)
        
        print("✅ Analysis complete!")
        return results
    
    def analyze_time_range(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze transcription within a specific time range.
        
        Args:
            start_time: Start time for analysis
            end_time: End time for analysis
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.entries:
            self.load_transcription()
        
        # Filter entries by time range
        filtered_entries = self.parser.get_content_by_time_range(
            self.entries, start_time, end_time
        )
        
        if not filtered_entries:
            raise ValueError("No transcription data found in specified time range")
        
        print(f"🔍 Analyzing {len(filtered_entries)} entries in time range...")
        
        content = self.parser.get_content_text(filtered_entries)
        
        results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "time_range": {
                    "start": start_time.isoformat() if start_time else None,
                    "end": end_time.isoformat() if end_time else None
                },
                "statistics": self.parser.get_statistics(filtered_entries)
            },
            "summary": self.openai_analyzer.summarize_content(content),
            "key_points": self.openai_analyzer.extract_key_points(content),
            "sentiment": self.openai_analyzer.analyze_sentiment(content),
        }
        
        if save_results:
            filename = f"analysis_range_{start_time or 'start'}_{end_time or 'end'}.json"
            self._save_analysis_results(results, filename)
        
        return results
    
    def custom_analysis(self, prompt: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Perform custom analysis with user-defined prompt.
        
        Args:
            prompt: Custom analysis prompt
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with analysis results
        """
        if not self.entries:
            self.load_transcription()
        
        if not self.entries:
            raise ValueError("No transcription data found")
        
        print(f"🔍 Running custom analysis: {prompt[:50]}...")
        
        full_content = self.parser.get_content_text(self.entries)
        
        # For large content, use chunking
        if len(full_content) > config.max_content_length:
            analysis_result = self._analyze_with_chunking(full_content, "custom", prompt)
        else:
            analysis_result = self.openai_analyzer.custom_analysis(full_content, prompt)
        
        results = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "custom_prompt": prompt,
                "statistics": self.parser.get_statistics(self.entries)
            },
            "analysis": analysis_result
        }
        
        if save_results:
            filename = f"custom_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_analysis_results(results, filename)
        
        return results
    
    def _analyze_with_chunking(
        self, 
        content: str, 
        analysis_type: str, 
        custom_prompt: Optional[str] = None
    ) -> Any:
        """
        Perform analysis with automatic chunking for large content.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis (summary, key_points, sentiment, custom)
            custom_prompt: Custom prompt for custom analysis
            
        Returns:
            Analysis result (type depends on analysis_type)
        """
        # If content is small enough, analyze directly
        if len(content) <= config.max_content_length:
            return self._perform_single_analysis(content, analysis_type, custom_prompt)
        
        # Split into chunks and analyze each
        print(f"📊 Content too large ({len(content)} chars), splitting into chunks...")
        
        # Create temporary entries for chunking
        temp_entries = [
            type('TempEntry', (), {'content': content, 'timestamp': datetime.now()})()
        ]
        
        # Use a simple text-based chunking approach
        chunks = []
        words = content.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > config.max_content_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"📊 Analyzing {len(chunks)} chunks...")
        
        # Analyze each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"📊 Processing chunk {i+1}/{len(chunks)}...")
            result = self._perform_single_analysis(chunk, analysis_type, custom_prompt)
            chunk_results.append(result)
        
        # Combine results based on analysis type
        return self._combine_chunk_results(chunk_results, analysis_type)
    
    def _perform_single_analysis(
        self, 
        content: str, 
        analysis_type: str, 
        custom_prompt: Optional[str] = None
    ) -> Any:
        """Perform a single analysis on content."""
        if analysis_type == "summary":
            return self.openai_analyzer.summarize_content(content)
        elif analysis_type == "key_points":
            return self.openai_analyzer.extract_key_points(content)
        elif analysis_type == "sentiment":
            return self.openai_analyzer.analyze_sentiment(content)
        elif analysis_type == "custom":
            return self.openai_analyzer.custom_analysis(content, custom_prompt)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def _combine_chunk_results(self, chunk_results: List[Any], analysis_type: str) -> Any:
        """Combine results from multiple chunks."""
        if analysis_type == "summary":
            # Combine summaries into a meta-summary
            combined_summaries = "\n\n".join(chunk_results)
            return self.openai_analyzer.summarize_content(
                combined_summaries, 
                max_length=1000
            )
        elif analysis_type == "key_points":
            # Flatten and deduplicate key points
            all_points = []
            for chunk_points in chunk_results:
                all_points.extend(chunk_points)
            return list(dict.fromkeys(all_points))  # Remove duplicates while preserving order
        elif analysis_type == "sentiment":
            # Average sentiment scores and combine emotions
            if not chunk_results:
                return {"sentiment": "neutral", "confidence": 0.5, "key_emotions": [], "explanation": "No data"}
            
            sentiments = [r.get("sentiment", "neutral") for r in chunk_results]
            confidences = [r.get("confidence", 0.5) for r in chunk_results]
            all_emotions = []
            for r in chunk_results:
                all_emotions.extend(r.get("key_emotions", []))
            
            # Most common sentiment
            sentiment_counts = {}
            for s in sentiments:
                sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
            most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            return {
                "sentiment": most_common_sentiment,
                "confidence": sum(confidences) / len(confidences),
                "key_emotions": list(dict.fromkeys(all_emotions)),
                "explanation": f"Combined analysis of {len(chunk_results)} chunks"
            }
        elif analysis_type == "custom":
            # Combine custom analysis results
            return "\n\n".join(chunk_results)
        else:
            return chunk_results
    
    def _save_analysis_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save analysis results to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_{timestamp}.json"
        
        output_dir = config.get_analysis_output_dir()
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Analysis results saved to: {output_path}")
        except Exception as e:
            print(f"⚠️  Failed to save analysis results: {e}")
