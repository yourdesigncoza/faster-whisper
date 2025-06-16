#!/usr/bin/env python3
"""
Transcription Analysis CLI Tool

Command-line interface for analyzing transcribed YouTube content using OpenAI.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analysis.analyzer import TranscriptionAnalyzer
from app.utils.config import config


def print_analysis_summary(results: dict):
    """Print a formatted summary of analysis results."""
    print("\n" + "="*60)
    print("📊 TRANSCRIPTION ANALYSIS RESULTS")
    print("="*60)
    
    # Metadata
    if "metadata" in results:
        meta = results["metadata"]
        print(f"\n📅 Analysis Date: {meta.get('analysis_timestamp', 'Unknown')}")
        
        if "statistics" in meta:
            stats = meta["statistics"]
            print(f"📈 Statistics:")
            print(f"   • Total Entries: {stats.get('total_entries', 0)}")
            print(f"   • Total Words: {stats.get('total_words', 0)}")
            print(f"   • Total Characters: {stats.get('total_characters', 0)}")
            print(f"   • Time Span: {stats.get('time_span', 'Unknown')}")
    
    # Summary
    if "summary" in results:
        print(f"\n📝 SUMMARY:")
        print("-" * 40)
        print(results["summary"])
    
    # Key Points
    if "key_points" in results:
        print(f"\n🔑 KEY POINTS:")
        print("-" * 40)
        for i, point in enumerate(results["key_points"], 1):
            print(f"{i:2d}. {point}")
    
    # Sentiment
    if "sentiment" in results:
        sentiment = results["sentiment"]
        print(f"\n😊 SENTIMENT ANALYSIS:")
        print("-" * 40)
        print(f"   • Overall Sentiment: {sentiment.get('sentiment', 'Unknown').title()}")
        print(f"   • Confidence: {sentiment.get('confidence', 0):.2f}")
        if sentiment.get('key_emotions'):
            print(f"   • Key Emotions: {', '.join(sentiment['key_emotions'])}")
        if sentiment.get('explanation'):
            print(f"   • Explanation: {sentiment['explanation']}")
    
    # Custom Analysis
    if "analysis" in results:
        print(f"\n🔍 CUSTOM ANALYSIS:")
        print("-" * 40)
        print(results["analysis"])
    
    print("\n" + "="*60)


def process_batch_directory(directory_path: str, args, base_analyzer) -> dict:
    """Process multiple transcription files in a directory."""
    directory = Path(directory_path)

    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory_path}")

    # Find transcription files (txt files)
    transcription_files = list(directory.glob("*.txt"))

    if not transcription_files:
        raise ValueError(f"No .txt files found in directory: {directory_path}")

    print(f"📁 Found {len(transcription_files)} transcription files")

    batch_results = {
        "batch_metadata": {
            "directory": str(directory),
            "total_files": len(transcription_files),
            "processed_files": 0,
            "failed_files": 0,
            "analysis_timestamp": datetime.now().isoformat()
        },
        "file_results": {},
        "batch_summary": {}
    }

    successful_analyses = []

    for i, file_path in enumerate(transcription_files, 1):
        try:
            print(f"\n📄 Processing file {i}/{len(transcription_files)}: {file_path.name}")

            # Create analyzer for this file
            analyzer = TranscriptionAnalyzer(file_path)
            analyzer.load_transcription()

            # Determine analysis type and perform analysis
            if args.full:
                result = analyzer.analyze_full_transcription(save_results=not args.no_save)
            elif args.summary:
                content = analyzer.parser.get_content_text(analyzer.entries)
                summary = analyzer.openai_analyzer.summarize_content(content)
                result = {"summary": summary}
            elif args.key_points:
                content = analyzer.parser.get_content_text(analyzer.entries)
                key_points = analyzer.openai_analyzer.extract_key_points(content)
                result = {"key_points": key_points}
            elif args.sentiment:
                content = analyzer.parser.get_content_text(analyzer.entries)
                sentiment = analyzer.openai_analyzer.analyze_sentiment(content)
                result = {"sentiment": sentiment}
            elif args.custom:
                result = analyzer.custom_analysis(args.custom, save_results=not args.no_save)

            # Store result
            batch_results["file_results"][file_path.name] = {
                "status": "success",
                "result": result,
                "statistics": analyzer.parser.get_statistics(analyzer.entries)
            }

            successful_analyses.append(result)
            batch_results["batch_metadata"]["processed_files"] += 1

            print(f"✅ Successfully processed {file_path.name}")

        except Exception as e:
            print(f"❌ Failed to process {file_path.name}: {e}")
            batch_results["file_results"][file_path.name] = {
                "status": "failed",
                "error": str(e)
            }
            batch_results["batch_metadata"]["failed_files"] += 1

    # Generate batch summary if we have successful analyses
    if successful_analyses and args.full:
        batch_results["batch_summary"] = generate_batch_summary(successful_analyses)

    # Save batch results
    if not args.no_save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_path = config.get_analysis_output_dir() / f"batch_analysis_{timestamp}.json"

        with open(batch_output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Batch results saved to: {batch_output_path}")

    return batch_results


def generate_batch_summary(analyses: list) -> dict:
    """Generate a summary across multiple analysis results."""
    if not analyses:
        return {}

    # Combine all summaries
    all_summaries = []
    all_key_points = []
    all_sentiments = []

    for analysis in analyses:
        if "summary" in analysis:
            all_summaries.append(analysis["summary"])
        if "key_points" in analysis:
            all_key_points.extend(analysis["key_points"])
        if "sentiment" in analysis:
            all_sentiments.append(analysis["sentiment"])

    summary = {}

    # Meta-summary of all summaries
    if all_summaries:
        combined_summaries = "\n\n".join(all_summaries)
        # Note: This would require an analyzer instance, so we'll just combine them
        summary["combined_summary"] = "Combined summaries from all files:\n\n" + combined_summaries

    # Deduplicated key points
    if all_key_points:
        unique_points = list(dict.fromkeys(all_key_points))  # Remove duplicates
        summary["combined_key_points"] = unique_points

    # Average sentiment
    if all_sentiments:
        sentiments = [s.get("sentiment", "neutral") for s in all_sentiments]
        confidences = [s.get("confidence", 0.5) for s in all_sentiments]

        sentiment_counts = {}
        for s in sentiments:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        avg_confidence = sum(confidences) / len(confidences)

        summary["combined_sentiment"] = {
            "overall_sentiment": most_common_sentiment,
            "average_confidence": avg_confidence,
            "sentiment_distribution": sentiment_counts
        }

    return summary


def print_batch_summary(batch_results: dict):
    """Print a summary of batch processing results."""
    print("\n" + "="*60)
    print("📦 BATCH PROCESSING RESULTS")
    print("="*60)

    meta = batch_results["batch_metadata"]
    print(f"\n📊 Processing Summary:")
    print(f"   • Total Files: {meta['total_files']}")
    print(f"   • Successfully Processed: {meta['processed_files']}")
    print(f"   • Failed: {meta['failed_files']}")
    print(f"   • Success Rate: {meta['processed_files']/meta['total_files']*100:.1f}%")

    if "batch_summary" in batch_results and batch_results["batch_summary"]:
        summary = batch_results["batch_summary"]

        if "combined_summary" in summary:
            print(f"\n📝 COMBINED SUMMARY:")
            print("-" * 40)
            print(summary["combined_summary"][:500] + "..." if len(summary["combined_summary"]) > 500 else summary["combined_summary"])

        if "combined_key_points" in summary:
            print(f"\n🔑 COMBINED KEY POINTS:")
            print("-" * 40)
            for i, point in enumerate(summary["combined_key_points"][:10], 1):  # Show top 10
                print(f"{i:2d}. {point}")
            if len(summary["combined_key_points"]) > 10:
                print(f"    ... and {len(summary['combined_key_points']) - 10} more")

        if "combined_sentiment" in summary:
            sentiment = summary["combined_sentiment"]
            print(f"\n😊 COMBINED SENTIMENT:")
            print("-" * 40)
            print(f"   • Overall Sentiment: {sentiment['overall_sentiment'].title()}")
            print(f"   • Average Confidence: {sentiment['average_confidence']:.2f}")
            print(f"   • Distribution: {sentiment['sentiment_distribution']}")

    print("\n" + "="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze transcribed YouTube content using OpenAI",
        epilog="""
Examples:
  %(prog)s --full                                    # Analyze full transcription
  %(prog)s --summary                                 # Generate summary only
  %(prog)s --key-points                              # Extract key points only
  %(prog)s --sentiment                               # Analyze sentiment only
  %(prog)s --custom "What are the main trading strategies mentioned?"
  %(prog)s --file custom_output.txt --full          # Analyze specific file
  %(prog)s --no-save --summary                      # Don't save results to file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    parser.add_argument("--file", "-f", type=str,
                       help="Transcription file to analyze (default: output.txt)")
    
    # Analysis type options
    analysis_group = parser.add_mutually_exclusive_group(required=True)
    analysis_group.add_argument("--full", action="store_true",
                               help="Perform full analysis (summary, key points, sentiment)")
    analysis_group.add_argument("--summary", action="store_true",
                               help="Generate summary only")
    analysis_group.add_argument("--key-points", action="store_true",
                               help="Extract key points only")
    analysis_group.add_argument("--sentiment", action="store_true",
                               help="Analyze sentiment only")
    analysis_group.add_argument("--custom", type=str,
                               help="Custom analysis with specified prompt")
    analysis_group.add_argument("--batch", type=str,
                               help="Batch process multiple files from directory")
    
    # Output options
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")
    parser.add_argument("--output", "-o", type=str,
                       help="Output file for results (JSON format)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress output (only save to file)")
    
    # Configuration options
    parser.add_argument("--model", type=str,
                       help="OpenAI model to use (default from config)")
    parser.add_argument("--max-tokens", type=int,
                       help="Maximum tokens for API calls (default from config)")
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        transcription_file = Path(args.file) if args.file else None
        analyzer = TranscriptionAnalyzer(transcription_file)
        
        # Override model if specified
        if args.model:
            analyzer.openai_analyzer.model = args.model
        
        # Load transcription
        if not args.quiet:
            print("📖 Loading transcription data...")
        
        analyzer.load_transcription()
        
        # Perform analysis based on selected type
        save_results = not args.no_save
        
        if args.full:
            if not args.quiet:
                print("🔍 Performing comprehensive analysis...")
            results = analyzer.analyze_full_transcription(save_results=save_results)
            
        elif args.summary:
            if not args.quiet:
                print("📝 Generating summary...")
            content = analyzer.parser.get_content_text(analyzer.entries)
            summary = analyzer.openai_analyzer.summarize_content(content)
            results = {
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "summary",
                    "statistics": analyzer.parser.get_statistics(analyzer.entries)
                },
                "summary": summary
            }
            
        elif args.key_points:
            if not args.quiet:
                print("🔑 Extracting key points...")
            content = analyzer.parser.get_content_text(analyzer.entries)
            key_points = analyzer.openai_analyzer.extract_key_points(content)
            results = {
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "key_points",
                    "statistics": analyzer.parser.get_statistics(analyzer.entries)
                },
                "key_points": key_points
            }
            
        elif args.sentiment:
            if not args.quiet:
                print("😊 Analyzing sentiment...")
            content = analyzer.parser.get_content_text(analyzer.entries)
            sentiment = analyzer.openai_analyzer.analyze_sentiment(content)
            results = {
                "metadata": {
                    "analysis_timestamp": datetime.now().isoformat(),
                    "analysis_type": "sentiment",
                    "statistics": analyzer.parser.get_statistics(analyzer.entries)
                },
                "sentiment": sentiment
            }
            
        elif args.custom:
            if not args.quiet:
                print(f"🔍 Running custom analysis...")
            results = analyzer.custom_analysis(args.custom, save_results=save_results)

        elif args.batch:
            if not args.quiet:
                print(f"📦 Processing batch directory: {args.batch}")
            results = process_batch_directory(args.batch, args, analyzer)

            # For batch processing, results is a summary
            if not args.quiet:
                print_batch_summary(results)
            return 0
        
        # Save results to custom output file if specified
        if args.output and not args.no_save:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            if not args.quiet:
                print(f"💾 Results saved to: {output_path}")
        
        # Display results unless quiet mode
        if not args.quiet:
            print_analysis_summary(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
