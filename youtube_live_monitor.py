#!/usr/bin/env python3
"""
YouTube Live Stream Trading Monitor

Continuously transcribes YouTube live streams and performs periodic analysis
to detect trading signals and opportunities. Integrates the existing YouTube
listener with intelligent incremental analysis.

Features:
- Continuous live stream transcription
- Incremental analysis every 3 minutes
- Trading signal detection with alerts
- Smart content tracking to avoid re-analysis
- Rolling context window for analysis continuity
"""

import argparse
import datetime
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import traceback

# Import existing components
from app.youtube_listener import EnhancedYouTubeListener, RetryConfig
from app.analysis.analyzer import TranscriptionAnalyzer
from app.analysis.trading_intent_detector import TradingIntentDetector
from app.utils.transcription_parser import TranscriptionEntry


class TradingSignalDetector:
    """Detects trading signals and intent from transcription analysis."""
    
    # Trading-related keywords and phrases
    ENTRY_SIGNALS = [
        "entry", "enter", "buy", "sell", "long", "short", "position",
        "trade setup", "setup", "signal", "breakout", "bounce",
        "support", "resistance", "trend", "reversal"
    ]
    
    INTENT_PHRASES = [
        "looking for", "waiting for", "watching", "ready to",
        "preparing", "setting up", "about to", "going to trade",
        "trade incoming", "get ready", "here we go", "this is it"
    ]
    
    CONFIRMATION_WORDS = [
        "confirmed", "triggered", "activated", "go", "now",
        "execute", "take it", "boom", "perfect", "there it is"
    ]
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_for_trading_signals(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the transcription analysis for trading signals.
        
        Args:
            analysis_result: Result from transcription analysis
            
        Returns:
            Dictionary with trading signal analysis
        """
        signal_data = {
            "trading_intent_detected": False,
            "signal_strength": 0,  # 0-10 scale
            "detected_signals": [],
            "alert_message": None,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Extract text content for analysis
            content_sources = []
            
            if "summary" in analysis_result:
                content_sources.append(("summary", analysis_result["summary"]))
            
            if "key_points" in analysis_result:
                if isinstance(analysis_result["key_points"], list):
                    content_sources.append(("key_points", " ".join(analysis_result["key_points"])))
                else:
                    content_sources.append(("key_points", str(analysis_result["key_points"])))
            
            if "analysis" in analysis_result:
                content_sources.append(("custom_analysis", str(analysis_result["analysis"])))
            
            # Analyze each content source
            total_signals = 0
            detected_signals = []
            
            for source_type, content in content_sources:
                if not content:
                    continue
                    
                content_lower = content.lower()
                source_signals = 0
                
                # Check for entry signals
                for signal in self.ENTRY_SIGNALS:
                    if signal in content_lower:
                        source_signals += 1
                        detected_signals.append(f"{signal} (in {source_type})")
                
                # Check for intent phrases (higher weight)
                for phrase in self.INTENT_PHRASES:
                    if phrase in content_lower:
                        source_signals += 2
                        detected_signals.append(f"INTENT: {phrase} (in {source_type})")
                
                # Check for confirmation words (highest weight)
                for word in self.CONFIRMATION_WORDS:
                    if word in content_lower:
                        source_signals += 3
                        detected_signals.append(f"CONFIRMATION: {word} (in {source_type})")
                
                total_signals += source_signals
            
            # Calculate signal strength (0-10 scale)
            signal_data["signal_strength"] = min(10, total_signals)
            signal_data["detected_signals"] = detected_signals
            
            # Determine if trading intent is detected
            if total_signals >= 3:  # Threshold for trading intent
                signal_data["trading_intent_detected"] = True
                
                # Generate alert message based on signal strength
                if total_signals >= 8:
                    signal_data["alert_message"] = "🚨 BOOM! GET READY FOR A TRADE! 🚨 High confidence signal detected!"
                elif total_signals >= 5:
                    signal_data["alert_message"] = "⚡ BOOM! GET READY FOR A TRADE! ⚡ Strong signal detected!"
                else:
                    signal_data["alert_message"] = "📈 Boom, Get Ready for a trade! Trading setup detected."
            
            self.logger.debug(f"Trading signal analysis: {total_signals} signals, strength {signal_data['signal_strength']}")
            
        except Exception as e:
            self.logger.error(f"Error in trading signal analysis: {e}")
            self.logger.debug(f"Trading signal error details: {traceback.format_exc()}")
        
        return signal_data


class IncrementalAnalysisTracker:
    """Tracks analysis progress and manages incremental content processing."""

    def __init__(self, transcript_file: Path):
        self.transcript_file = transcript_file
        self.last_analyzed_position = 0
        self.last_analysis_time = None
        self.analysis_history = []
        self.trading_context = {
            "last_intent_detected": False,
            "last_intent_time": None,
            "recent_signals": [],
            "market_sentiment": "neutral"
        }
        self.logger = logging.getLogger(__name__)
    
    def get_new_content(self) -> Tuple[List[TranscriptionEntry], int]:
        """
        Get new transcript content since last analysis.
        
        Returns:
            Tuple of (new_entries, new_position)
        """
        try:
            if not self.transcript_file.exists():
                return [], 0
            
            # Read all entries
            from app.utils.transcription_parser import TranscriptionParser
            parser = TranscriptionParser(self.transcript_file)
            all_entries = parser.read_transcription_file()
            
            # Get entries after last analyzed position
            new_entries = all_entries[self.last_analyzed_position:]
            new_position = len(all_entries)
            
            self.logger.debug(f"Found {len(new_entries)} new entries (position {self.last_analyzed_position} -> {new_position})")
            
            return new_entries, new_position
            
        except Exception as e:
            self.logger.error(f"Error getting new content: {e}")
            return [], self.last_analyzed_position
    
    def update_position(self, new_position: int):
        """Update the last analyzed position."""
        self.last_analyzed_position = new_position
        self.last_analysis_time = datetime.datetime.now()
        self.logger.debug(f"Updated analysis position to {new_position}")
    
    def add_to_history(self, analysis_result: Dict[str, Any]):
        """Add analysis result to history, maintaining rolling window."""
        self.analysis_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "result": analysis_result
        })

        # Update trading context based on this analysis
        self.update_trading_context(analysis_result)

        # Keep only last 10 analysis cycles for context
        if len(self.analysis_history) > 10:
            self.analysis_history = self.analysis_history[-10:]
    
    def get_context_summary(self) -> str:
        """Get a trading-focused summary of recent analysis for context."""
        context_parts = []

        # Add current trading context state
        if self.trading_context["last_intent_detected"]:
            context_parts.append(f"Previous intent detected at {self.trading_context['last_intent_time']}")
        else:
            context_parts.append("No recent trading intent detected")

        # Add recent signal information
        if self.trading_context["recent_signals"]:
            recent_signals = self.trading_context["recent_signals"][-3:]  # Last 3 signals
            context_parts.append(f"Recent signals: {', '.join(recent_signals)}")

        # Add market sentiment
        context_parts.append(f"Market sentiment: {self.trading_context['market_sentiment']}")

        # Add brief summary from recent analyses
        if self.analysis_history:
            recent_analysis = self.analysis_history[-1]["result"]
            if "trading_signals" in recent_analysis:
                signals = recent_analysis["trading_signals"]
                if signals.get("trading_intent_detected"):
                    context_parts.append("Last analysis: Trading intent was detected")
                else:
                    context_parts.append("Last analysis: No trading intent detected")

        return " | ".join(context_parts)

    def update_trading_context(self, analysis_result: Dict[str, Any]):
        """Update trading context based on analysis results."""
        if "trading_signals" in analysis_result:
            signals = analysis_result["trading_signals"]

            # Update intent detection status
            if signals.get("trading_intent_detected"):
                self.trading_context["last_intent_detected"] = True
                self.trading_context["last_intent_time"] = datetime.datetime.now().strftime("%H:%M:%S")

                # Add to recent signals
                if "intent_details" in signals:
                    details = signals["intent_details"]
                    signal_desc = f"{details.get('direction', 'unknown')} {details.get('instrument', 'position')}"
                    self.trading_context["recent_signals"].append(signal_desc)

                    # Keep only last 5 signals
                    if len(self.trading_context["recent_signals"]) > 5:
                        self.trading_context["recent_signals"] = self.trading_context["recent_signals"][-5:]
            else:
                self.trading_context["last_intent_detected"] = False

            # Update market sentiment based on analysis
            if "sentiment" in analysis_result:
                sentiment = analysis_result["sentiment"]
                if isinstance(sentiment, dict) and "overall_sentiment" in sentiment:
                    self.trading_context["market_sentiment"] = sentiment["overall_sentiment"]


class YouTubeLiveMonitor:
    """Main class for YouTube live stream monitoring and analysis."""
    
    def __init__(
        self,
        youtube_url: str,
        analysis_interval: float = 180.0,  # 3 minutes
        model_size: str = "small",
        max_retries: int = 10,
        log_level: str = "DEBUG"
    ):
        """
        Initialize the YouTube live monitor.
        
        Args:
            youtube_url: YouTube live stream URL
            analysis_interval: Seconds between analysis cycles (default 3 minutes)
            model_size: Whisper model size (default "small")
            max_retries: Maximum retry attempts (default 10)
            log_level: Logging level (default "DEBUG")
        """
        self.youtube_url = youtube_url
        self.analysis_interval = analysis_interval
        self.model_size = model_size
        self.max_retries = max_retries
        self.log_level = log_level
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.transcript_file = Path("/home/laudes/zoot/projects/faster-whisper/analysis_results/youtube/transcript.txt")
        self.youtube_listener = None
        self.analyzer = None
        self.tracker = IncrementalAnalysisTracker(self.transcript_file)
        self.signal_detector = TradingSignalDetector()  # Keep old detector for comparison
        self.intent_detector = None  # Will be initialized when analyzer is ready
        
        # Control flags
        self.is_running = False
        self.should_stop = False
        
        # Threads
        self.analysis_thread = None
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.log_level.upper()))
        
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
        self.logger.info("🛑 Received shutdown signal, stopping monitor...")
        self.stop()
        sys.exit(0)

    def _initialize_components(self):
        """Initialize YouTube listener and analyzer components."""
        try:
            # Create retry configuration
            retry_config = RetryConfig(
                max_retries=self.max_retries,
                initial_delay=1.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
                jitter=True
            )

            # Initialize YouTube listener
            self.youtube_listener = EnhancedYouTubeListener(
                youtube_url=self.youtube_url,
                model_size=self.model_size,
                device="cpu",
                compute_type="int8",
                output_file=str(self.transcript_file),
                chunk_duration=30.0,
                language=None,
                retry_config=retry_config,
                log_level=self.log_level,
                use_app_config=True
            )

            # Initialize analyzer
            self.analyzer = TranscriptionAnalyzer(self.transcript_file)

            # Initialize intent detector with the analyzer's OpenAI client
            self.intent_detector = TradingIntentDetector(self.analyzer.openai_analyzer)

            self.logger.info("✅ Components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise

    def _analysis_worker(self):
        """Worker thread for periodic analysis."""
        self.logger.info(f"🔍 Starting analysis worker (interval: {self.analysis_interval}s)")

        while self.is_running and not self.should_stop:
            try:
                # Wait for analysis interval
                time.sleep(self.analysis_interval)

                if self.should_stop:
                    break

                # Perform incremental analysis
                self._perform_incremental_analysis()

            except Exception as e:
                self.logger.error(f"❌ Analysis worker error: {e}")
                self.logger.debug(f"Analysis worker error details: {traceback.format_exc()}")
                time.sleep(30)  # Wait before retrying

        self.logger.info("🛑 Analysis worker stopped")

    def _perform_incremental_analysis(self):
        """Perform analysis on new transcript content."""
        try:
            # Get new content since last analysis
            new_entries, new_position = self.tracker.get_new_content()

            if not new_entries:
                self.logger.debug("📭 No new content to analyze")
                return

            self.logger.info(f"🔍 Analyzing {len(new_entries)} new transcript entries...")

            # Create temporary analyzer for new content
            temp_transcript_content = []
            for entry in new_entries:
                temp_transcript_content.append(f"[{entry.timestamp}] {entry.content}")

            new_content_text = "\n".join(temp_transcript_content)

            if len(new_content_text.strip()) < 10:
                self.logger.debug("📭 New content too short for meaningful analysis")
                self.tracker.update_position(new_position)
                return

            # Get context from previous analyses
            context = self.tracker.get_context_summary()

            # Perform analysis with context
            analysis_prompt = f"""
            Analyze this new transcript content for trading signals and opportunities.

            Previous context:
            {context}

            New content to analyze:
            {new_content_text}

            Focus on:
            1. Trading setups, signals, or opportunities mentioned
            2. Market analysis and trader sentiment
            3. Entry/exit points or trade preparation
            4. Any indication the trader is about to make a trade

            Provide a concise analysis highlighting any trading-relevant information.
            """

            # Perform custom analysis
            analysis_result = self.analyzer.custom_analysis(
                prompt=analysis_prompt,
                save_results=False
            )

            # Add metadata
            analysis_result["incremental_analysis"] = True
            analysis_result["entries_analyzed"] = len(new_entries)
            analysis_result["content_length"] = len(new_content_text)

            # Detect trading signals using both methods for comparison
            # Old keyword-based method
            old_signal_analysis = self.signal_detector.analyze_for_trading_signals(analysis_result)

            # New intent-focused method
            intent = self.intent_detector.detect_intent(new_content_text, context)
            new_signal_analysis = self.intent_detector.create_signal_analysis(intent)

            # Use new method as primary, but log comparison
            analysis_result["trading_signals"] = new_signal_analysis
            analysis_result["trading_signals_comparison"] = {
                "old_method": old_signal_analysis,
                "new_method": new_signal_analysis,
                "methods_agree": old_signal_analysis["trading_intent_detected"] == new_signal_analysis["trading_intent_detected"]
            }

            # Log analysis results
            self._log_analysis_results(analysis_result, new_signal_analysis)

            # Save analysis if trading signals detected
            if new_signal_analysis["trading_intent_detected"]:
                self._save_signal_analysis(analysis_result)

            # Update tracker
            self.tracker.add_to_history(analysis_result)
            self.tracker.update_position(new_position)

        except Exception as e:
            self.logger.error(f"❌ Incremental analysis error: {e}")
            self.logger.debug(f"Incremental analysis error details: {traceback.format_exc()}")

    def _log_analysis_results(self, analysis_result: Dict[str, Any], signal_analysis: Dict[str, Any]):
        """Log analysis results and trading signals."""
        try:
            # Log basic analysis info
            entries_count = analysis_result.get("entries_analyzed", 0)
            content_length = analysis_result.get("content_length", 0)

            self.logger.info(f"📊 Analysis complete: {entries_count} entries, {content_length} chars")

            # Log trading signal results
            if signal_analysis["trading_intent_detected"]:
                strength = signal_analysis["signal_strength"]
                signals = signal_analysis["detected_signals"]
                alert = signal_analysis["alert_message"]

                self.logger.warning(f"🚨 TRADING SIGNAL DETECTED! Strength: {strength}/10")
                self.logger.warning(f"🎯 {alert}")
                self.logger.info(f"📈 Detected signals: {', '.join(signals[:5])}")  # Show first 5

                # Print prominent alert to console
                print("\n" + "="*60)
                print(f"🚨 {alert} 🚨")
                print(f"Signal Strength: {strength}/10")
                print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*60 + "\n")

            else:
                self.logger.debug(f"📉 No trading signals detected (strength: {signal_analysis['signal_strength']}/10)")

            # Log analysis content (debug level)
            if "analysis" in analysis_result:
                analysis_text = str(analysis_result["analysis"])[:200]
                self.logger.debug(f"📝 Analysis excerpt: {analysis_text}...")

        except Exception as e:
            self.logger.error(f"❌ Error logging analysis results: {e}")

    def _save_signal_analysis(self, analysis_result: Dict[str, Any]):
        """Save analysis results when trading signals are detected."""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_signal_{timestamp}.json"

            output_dir = Path("/home/laudes/zoot/projects/faster-whisper/analysis_results")
            output_path = output_dir / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"💾 Trading signal analysis saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save signal analysis: {e}")

    def start(self):
        """Start the YouTube live monitor."""
        try:
            self.logger.info("🚀 Starting YouTube Live Trading Monitor...")
            self.logger.info(f"🎵 YouTube URL: {self.youtube_url}")
            self.logger.info(f"📁 Transcript file: {self.transcript_file}")
            self.logger.info(f"⏱️  Analysis interval: {self.analysis_interval}s")
            self.logger.info(f"🤖 Model: {self.model_size}, Max retries: {self.max_retries}")
            self.logger.info("💡 Monitor will analyze new content every 3 minutes for trading signals")
            self.logger.info("-" * 70)

            # Initialize components
            self._initialize_components()

            # Clear transcript file for fresh start
            if self.transcript_file.exists():
                self.transcript_file.unlink()
            self.transcript_file.touch()

            # Start monitoring
            self.is_running = True
            self.should_stop = False

            # Start analysis worker thread
            self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
            self.analysis_thread.start()

            # Start YouTube listener (this will block)
            self.logger.info("🎵 Starting YouTube transcription...")
            self.youtube_listener.start()

        except KeyboardInterrupt:
            self.logger.info("🛑 Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"❌ Failed to start monitor: {e}")
            self.logger.debug(f"Startup error details: {traceback.format_exc()}")
            raise
        finally:
            self.stop()

    def stop(self):
        """Stop the YouTube live monitor."""
        if not self.is_running:
            return

        self.logger.info("🛑 Stopping YouTube Live Trading Monitor...")

        self.should_stop = True
        self.is_running = False

        # Stop YouTube listener
        if self.youtube_listener:
            try:
                self.youtube_listener.stop()
            except Exception as e:
                self.logger.warning(f"⚠️  Error stopping YouTube listener: {e}")

        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.logger.info("⏳ Stopping analysis worker...")
            self.analysis_thread.join(timeout=10)
            if self.analysis_thread.is_alive():
                self.logger.warning("⚠️  Analysis thread did not stop gracefully")

        self.logger.info("✅ YouTube Live Trading Monitor stopped successfully!")

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        return {
            "is_running": self.is_running,
            "youtube_url": self.youtube_url,
            "analysis_interval": self.analysis_interval,
            "model_size": self.model_size,
            "transcript_file": str(self.transcript_file),
            "last_analysis_time": self.tracker.last_analysis_time.isoformat() if self.tracker.last_analysis_time else None,
            "last_analyzed_position": self.tracker.last_analyzed_position,
            "analysis_history_count": len(self.tracker.analysis_history),
            "youtube_listener_status": self.youtube_listener.get_status() if self.youtube_listener else None
        }


def main():
    """Main entry point for the YouTube Live Trading Monitor."""
    parser = argparse.ArgumentParser(
        description="YouTube Live Stream Trading Monitor - Continuous transcription with trading signal detection",
        epilog="""
Examples:
  %(prog)s --url "https://www.youtube.com/watch?v=LIVE_STREAM_ID"
  %(prog)s --url "https://youtu.be/LIVE_STREAM_ID" --interval 120
  %(prog)s --url "https://www.youtube.com/watch?v=LIVE_STREAM_ID" --model base
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument("--url", required=False,
                       help="YouTube live stream URL to monitor")

    # Optional arguments with defaults
    parser.add_argument("--interval", type=float, default=180.0,
                       help="Analysis interval in seconds (default: 180 = 3 minutes)")
    parser.add_argument("--model", default="small",
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Whisper model size (default: small)")
    parser.add_argument("--max-retries", type=int, default=10,
                       help="Maximum retry attempts (default: 10)")
    parser.add_argument("--log-level", default="DEBUG",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: DEBUG)")

    args = parser.parse_args()

    # Get YouTube URL from user if not provided
    youtube_url = args.url
    if not youtube_url:
        print("🎵 YouTube Live Stream Trading Monitor")
        print("Features: Continuous transcription + Trading signal detection")
        print()
        print("This monitor will:")
        print("  • Transcribe live YouTube streams in real-time")
        print("  • Analyze new content every 3 minutes for trading signals")
        print("  • Alert with 'Boom, Get Ready for a trade!' when signals detected")
        print("  • Use incremental analysis to avoid re-processing content")
        print()
        print("Please enter a YouTube live stream URL:")
        youtube_url = input("URL: ").strip()

        if not youtube_url:
            print("❌ No URL provided. Exiting.")
            sys.exit(1)

    # Create and start monitor
    monitor = YouTubeLiveMonitor(
        youtube_url=youtube_url,
        analysis_interval=args.interval,
        model_size=args.model,
        max_retries=args.max_retries,
        log_level=args.log_level
    )

    try:
        monitor.start()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
