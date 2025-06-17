"""
Signal Detection Validation Framework

This module provides tools to compare the old keyword-based trading signal detection
with the new intent-focused LLM approach on historical transcript data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from .trading_intent_detector import TradingIntentDetector
from .analyzer import TranscriptionAnalyzer
from ..utils.transcription_parser import TranscriptionParser


@dataclass
class ValidationResult:
    """Results from comparing detection methods."""
    transcript_file: str
    total_segments: int
    old_method_detections: int
    new_method_detections: int
    agreements: int
    disagreements: int
    false_positive_reduction: float
    detailed_results: List[Dict[str, Any]]


class SignalDetectionValidator:
    """
    Validates and compares trading signal detection methods.
    
    This class helps evaluate the effectiveness of the new intent-focused
    approach compared to the old keyword-based method.
    """
    
    def __init__(self, analyzer: Optional[TranscriptionAnalyzer] = None):
        """
        Initialize the validator.
        
        Args:
            analyzer: TranscriptionAnalyzer instance. If None, creates a new one.
        """
        self.analyzer = analyzer
        self.intent_detector = None
        self.logger = logging.getLogger(__name__)
        
        # Import the old detector class
        from youtube_live_monitor import TradingSignalDetector
        self.old_detector = TradingSignalDetector()
    
    def validate_on_transcript(
        self, 
        transcript_file: Path, 
        segment_size: int = 5,
        save_results: bool = True
    ) -> ValidationResult:
        """
        Validate both detection methods on a transcript file.
        
        Args:
            transcript_file: Path to transcript file
            segment_size: Number of transcript entries per analysis segment
            save_results: Whether to save detailed results to file
            
        Returns:
            ValidationResult with comparison metrics
        """
        self.logger.info(f"🔍 Starting validation on {transcript_file}")
        
        # Initialize components if needed
        if not self.analyzer:
            self.analyzer = TranscriptionAnalyzer(transcript_file)
        
        if not self.intent_detector:
            self.intent_detector = TradingIntentDetector(self.analyzer.openai_analyzer)
        
        # Parse transcript
        parser = TranscriptionParser(transcript_file)
        entries = parser.read_transcription_file()
        
        if not entries:
            raise ValueError(f"No entries found in {transcript_file}")
        
        self.logger.info(f"📊 Analyzing {len(entries)} transcript entries in segments of {segment_size}")
        
        # Process in segments
        detailed_results = []
        old_detections = 0
        new_detections = 0
        agreements = 0
        
        for i in range(0, len(entries), segment_size):
            segment = entries[i:i + segment_size]
            segment_text = self._create_segment_text(segment)
            
            if len(segment_text.strip()) < 20:  # Skip very short segments
                continue
            
            # Test both methods
            old_result = self._test_old_method(segment_text)
            new_result = self._test_new_method(segment_text)
            
            # Record results
            segment_result = {
                "segment_index": i // segment_size,
                "segment_text": segment_text,
                "old_method": old_result,
                "new_method": new_result,
                "methods_agree": old_result["detected"] == new_result["detected"],
                "timestamp": datetime.now().isoformat()
            }
            
            detailed_results.append(segment_result)
            
            # Update counters
            if old_result["detected"]:
                old_detections += 1
            if new_result["detected"]:
                new_detections += 1
            if segment_result["methods_agree"]:
                agreements += 1
            
            # Log progress
            if (i // segment_size) % 10 == 0:
                self.logger.debug(f"Processed {i // segment_size + 1} segments...")
        
        # Calculate metrics
        total_segments = len(detailed_results)
        disagreements = total_segments - agreements
        
        # Calculate false positive reduction (assuming new method is more accurate)
        if old_detections > 0:
            false_positive_reduction = max(0, (old_detections - new_detections) / old_detections * 100)
        else:
            false_positive_reduction = 0.0
        
        result = ValidationResult(
            transcript_file=str(transcript_file),
            total_segments=total_segments,
            old_method_detections=old_detections,
            new_method_detections=new_detections,
            agreements=agreements,
            disagreements=disagreements,
            false_positive_reduction=false_positive_reduction,
            detailed_results=detailed_results
        )
        
        # Save results if requested
        if save_results:
            self._save_validation_results(result)
        
        self.logger.info(f"✅ Validation complete: {agreements}/{total_segments} agreements")
        return result
    
    def _create_segment_text(self, entries: List) -> str:
        """Create text from transcript entries."""
        lines = []
        for entry in entries:
            lines.append(f"[{entry.timestamp}] {entry.content}")
        return "\n".join(lines)
    
    def _test_old_method(self, segment_text: str) -> Dict[str, Any]:
        """Test the old keyword-based method."""
        try:
            # Create a mock analysis result for the old detector
            mock_analysis = {"analysis": segment_text}
            signal_analysis = self.old_detector.analyze_for_trading_signals(mock_analysis)
            
            return {
                "detected": signal_analysis["trading_intent_detected"],
                "strength": signal_analysis["signal_strength"],
                "signals": signal_analysis["detected_signals"],
                "method": "keyword_based"
            }
        except Exception as e:
            self.logger.error(f"Error in old method test: {e}")
            return {"detected": False, "strength": 0, "signals": [], "method": "keyword_based", "error": str(e)}
    
    def _test_new_method(self, segment_text: str) -> Dict[str, Any]:
        """Test the new intent-focused method."""
        try:
            intent = self.intent_detector.detect_intent(segment_text)
            
            return {
                "detected": intent.intent_detected,
                "confidence": intent.confidence,
                "direction": intent.direction,
                "instrument": intent.instrument,
                "entry_condition": intent.entry_condition,
                "method": "intent_focused"
            }
        except Exception as e:
            self.logger.error(f"Error in new method test: {e}")
            return {"detected": False, "confidence": 0.0, "method": "intent_focused", "error": str(e)}
    
    def _save_validation_results(self, result: ValidationResult):
        """Save validation results to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"signal_validation_{timestamp}.json"
            
            output_dir = Path("/home/laudes/zoot/projects/faster-whisper/analysis_results")
            output_path = output_dir / filename
            
            # Convert dataclass to dict for JSON serialization
            result_dict = {
                "transcript_file": result.transcript_file,
                "total_segments": result.total_segments,
                "old_method_detections": result.old_method_detections,
                "new_method_detections": result.new_method_detections,
                "agreements": result.agreements,
                "disagreements": result.disagreements,
                "false_positive_reduction": result.false_positive_reduction,
                "agreement_rate": result.agreements / result.total_segments * 100 if result.total_segments > 0 else 0,
                "detailed_results": result.detailed_results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"💾 Validation results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save validation results: {e}")
    
    def print_summary(self, result: ValidationResult):
        """Print a summary of validation results."""
        print("\n" + "="*60)
        print("🔍 SIGNAL DETECTION VALIDATION SUMMARY")
        print("="*60)
        print(f"📁 Transcript: {Path(result.transcript_file).name}")
        print(f"📊 Total segments analyzed: {result.total_segments}")
        print()
        print("🔍 Detection Results:")
        print(f"  • Old method (keyword-based): {result.old_method_detections} detections")
        print(f"  • New method (intent-focused): {result.new_method_detections} detections")
        print()
        print("📈 Agreement Analysis:")
        print(f"  • Methods agree: {result.agreements}/{result.total_segments} ({result.agreements/result.total_segments*100:.1f}%)")
        print(f"  • Methods disagree: {result.disagreements}/{result.total_segments} ({result.disagreements/result.total_segments*100:.1f}%)")
        print()
        print("🎯 Performance Metrics:")
        print(f"  • False positive reduction: {result.false_positive_reduction:.1f}%")
        print(f"  • Detection rate change: {((result.new_method_detections - result.old_method_detections) / max(1, result.old_method_detections) * 100):+.1f}%")
        print("="*60 + "\n")


def main():
    """Main function for running validation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate trading signal detection methods")
    parser.add_argument("--transcript", required=True, help="Path to transcript file")
    parser.add_argument("--segment-size", type=int, default=5, help="Entries per segment (default: 5)")
    parser.add_argument("--no-save", action="store_true", help="Don't save detailed results")
    
    args = parser.parse_args()
    
    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"❌ Transcript file not found: {transcript_path}")
        return
    
    validator = SignalDetectionValidator()
    result = validator.validate_on_transcript(
        transcript_path, 
        segment_size=args.segment_size,
        save_results=not args.no_save
    )
    
    validator.print_summary(result)


if __name__ == "__main__":
    main()
