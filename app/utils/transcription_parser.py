"""
Transcription parser for processing timestamped transcription data.

Handles reading and parsing transcription files from the YouTube listener.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .config import config


@dataclass
class TranscriptionEntry:
    """Represents a single transcription entry with timestamp and content."""
    timestamp: datetime
    content: str
    raw_line: str
    
    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.content}"


class TranscriptionParser:
    """Parser for timestamped transcription files."""
    
    def __init__(self, file_path: Optional[Path] = None):
        """
        Initialize transcription parser.
        
        Args:
            file_path: Path to transcription file (uses config default if not provided)
        """
        self.file_path = file_path or config.get_transcription_path()
        
        # Regex pattern for parsing timestamped lines
        # Matches format: [2024-01-15 14:23:45] transcription content
        self.timestamp_pattern = re.compile(
            r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(.+)$'
        )
    
    def read_transcription_file(self) -> List[TranscriptionEntry]:
        """
        Read and parse the transcription file.
        
        Returns:
            List of TranscriptionEntry objects
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Transcription file not found: {self.file_path}")
        
        entries = []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = self._parse_line(line, line_num)
                    if entry:
                        entries.append(entry)
            
            print(f"✅ Parsed {len(entries)} transcription entries from {self.file_path}")
            return entries
            
        except Exception as e:
            raise Exception(f"Error reading transcription file: {e}")
    
    def _parse_line(self, line: str, line_num: int) -> Optional[TranscriptionEntry]:
        """
        Parse a single line from the transcription file.
        
        Args:
            line: Line to parse
            line_num: Line number for error reporting
            
        Returns:
            TranscriptionEntry if line is valid, None otherwise
        """
        match = self.timestamp_pattern.match(line)
        if not match:
            print(f"⚠️  Skipping invalid line {line_num}: {line[:50]}...")
            return None
        
        timestamp_str, content = match.groups()
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            return TranscriptionEntry(
                timestamp=timestamp,
                content=content.strip(),
                raw_line=line
            )
        except ValueError as e:
            print(f"⚠️  Invalid timestamp on line {line_num}: {timestamp_str} - {e}")
            return None
    
    def get_content_by_time_range(
        self, 
        entries: List[TranscriptionEntry], 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TranscriptionEntry]:
        """
        Filter entries by time range.
        
        Args:
            entries: List of transcription entries
            start_time: Start time filter (inclusive)
            end_time: End time filter (inclusive)
            
        Returns:
            Filtered list of entries
        """
        filtered = entries
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        return filtered
    
    def get_content_text(self, entries: List[TranscriptionEntry]) -> str:
        """
        Extract plain text content from entries.
        
        Args:
            entries: List of transcription entries
            
        Returns:
            Combined text content
        """
        return ' '.join(entry.content for entry in entries)
    
    def get_content_with_timestamps(self, entries: List[TranscriptionEntry]) -> str:
        """
        Get content with timestamp information preserved.
        
        Args:
            entries: List of transcription entries
            
        Returns:
            Formatted text with timestamps
        """
        return '\n'.join(str(entry) for entry in entries)
    
    def split_into_chunks(
        self, 
        entries: List[TranscriptionEntry], 
        max_length: int = None
    ) -> List[List[TranscriptionEntry]]:
        """
        Split entries into chunks based on content length.
        
        Args:
            entries: List of transcription entries
            max_length: Maximum character length per chunk (uses config if not provided)
            
        Returns:
            List of entry chunks
        """
        if max_length is None:
            max_length = config.max_content_length
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for entry in entries:
            entry_length = len(entry.content)
            
            # If adding this entry would exceed max_length, start a new chunk
            if current_length + entry_length > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [entry]
                current_length = entry_length
            else:
                current_chunk.append(entry)
                current_length += entry_length
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_statistics(self, entries: List[TranscriptionEntry]) -> Dict[str, any]:
        """
        Get statistics about the transcription data.
        
        Args:
            entries: List of transcription entries
            
        Returns:
            Dictionary with statistics
        """
        if not entries:
            return {
                "total_entries": 0,
                "total_characters": 0,
                "total_words": 0,
                "time_span": "N/A",
                "first_timestamp": None,
                "last_timestamp": None
            }
        
        total_characters = sum(len(entry.content) for entry in entries)
        total_words = sum(len(entry.content.split()) for entry in entries)
        
        first_timestamp = min(entry.timestamp for entry in entries)
        last_timestamp = max(entry.timestamp for entry in entries)
        time_span = last_timestamp - first_timestamp
        
        return {
            "total_entries": len(entries),
            "total_characters": total_characters,
            "total_words": total_words,
            "time_span": str(time_span),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "average_entry_length": total_characters / len(entries) if entries else 0
        }
