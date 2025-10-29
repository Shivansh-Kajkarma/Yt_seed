"""
Client's Frequency-Based Competitor Discovery
==============================================
Pure frequency approach: Search video titles, count channel appearances
WITH DEBUG OUTPUTS AT EVERY STEP!
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
from utils.youtube_utils import (
    extract_channel_id,
    frequency_search_by_titles,
    enrich_channel_metadata_batch,
    filter_by_frequency_threshold
)


BASE = Path(__file__).resolve().parent

# Create debug output directory
DEBUG_DIR = BASE / "debug_outputs"
DEBUG_DIR.mkdir(exist_ok=True)


def save_debug_json(data, filename, description=""):
    """Save debug data as JSON with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DEBUG_DIR / f"{timestamp}_{filename}"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"   💾 Debug saved: {output_file.name}")
    if description:
        print(f"      {description}")
    
    return output_file


def save_debug_csv(df, filename, description=""):
    """Save debug data as CSV with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DEBUG_DIR / f"{timestamp}_{filename}"
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"   💾 Debug saved: {output_file.name}")
    if description:
        print(f"      {description}")
    
    return output_file


def main():
    """
    Client's approach implementation with debug outputs at every step
    """
    
    print("="*70)
    print("CLIENT'S FREQUENCY-BASED COMPETITOR DISCOVERY")
    print("="*70)
    print(f"Debug outputs will be saved to: {DEBUG_DIR}")
    
    # ============================================
    # STEP 1: Load Input Data
    # ============================================
    input_csv = BASE / "sample_videos.csv"
    
    if not input_csv.exists():
        print(f"❌ ERROR: {input_csv} not found!")
        print("   Run gt_videos.py first to generate sample_videos.csv")
        return
    
    print(f"\n📥 Loading data from {input_csv.name}...")
    df = pd.read_csv(input_csv)
    
    # Validate columns
    required_cols = ['Channel_Name', 'Channel_ID', 'title']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ ERROR: CSV must have columns: {required_cols}")
        return
    
    # Get seed channel info
    seed_channel_name = df['Channel_Name'].iloc[0]
    seed_channel_id = df['Channel_ID'].iloc[0]
    
    print(f"   Seed: {seed_channel_name}")
    print(f"   Channel ID: {seed_channel_id}")
    print(f"   Videos: {len(df)}")
    
    # Extract titles
    titles = df['title'].tolist()
    
    print(f"\n📋 Video titles to search:")
    for i, title in enumerate(titles[:5], 1):
        print(f"   {i}. {title[:70]}...")
    if len(titles) > 5:
        print(f"   ... and {len(titles) - 5} more")
    
    # DEBUG OUTPUT 1: Save input data
    print(f"\n📦 DEBUG OUTPUT 1: Input Data")
    debug_input = {
        "seed_channel_name": seed_channel_name,
        "seed_channel_id": seed_channel_id,
        "total_videos": len(titles),
        "video_titles": titles
    }
    save_debug_json(debug_input, "01_input_data.json", 
                   f"Seed info and {len(titles)} video titles")
    
    
    # ============================================
    # STEP 2: Frequency Search
    # ============================================
    print(f"\n{'='*70}")
    print("PHASE 1: FREQUENCY-BASED DISCOVERY")
    print(f"{'='*70}")
    
    channel_frequency, channel_metadata = frequency_search_by_titles(
        titles=titles,
        seed_channel_id=seed_channel_id,
        max_results_per_title=20  # 20 results per title
    )
    
    if not channel_frequency:
        print("❌ No channels found! Check API key or network.")
        return
    
    # Show top channels by frequency
    print(f"\n🔝 Top 15 channels by frequency:")
    for i, (ch_id, freq) in enumerate(channel_frequency.most_common(15), 1):
        name = channel_metadata.get(ch_id, {}).get("name", ch_id)
        pct = (freq / len(titles)) * 100
        print(f"   {i:2d}. {name:40s} {freq:2d}/{len(titles)} ({pct:5.1f}%)")
    
    # DEBUG OUTPUT 2: Save raw frequency data
    print(f"\n📦 DEBUG OUTPUT 2: Raw Frequency Data")
    debug_frequency = {
        "total_unique_channels": len(channel_frequency),
        "total_searches": len(titles),
        "frequency_data": [
            {
                "rank": i,
                "channel_id": ch_id,
                "channel_name": channel_metadata.get(ch_id, {}).get("name", "Unknown"),
                "frequency": freq,
                "percentage": round((freq / len(titles)) * 100, 1),
                "first_seen_in": channel_metadata.get(ch_id, {}).get("first_seen_in", "")
            }
            for i, (ch_id, freq) in enumerate(channel_frequency.most_common(), 1)
        ]
    }
    save_debug_json(debug_frequency, "02_raw_frequency.json",
                   f"{len(channel_frequency)} unique channels with frequency counts")
    
    # Also save as CSV for easy viewing
    df_freq = pd.DataFrame(debug_frequency["frequency_data"])
    save_debug_csv(df_freq, "02_raw_frequency.csv",
                  "Raw frequency data in CSV format")
    
    
    # ============================================
    # STEP 3: Enrich Metadata (Batch!)
    # ============================================
    print(f"\n{'='*70}")
    print("PHASE 2: ENRICHING METADATA")
    print(f"{'='*70}")
    
    channel_ids = list(channel_frequency.keys())
    enriched_metadata = enrich_channel_metadata_batch(
        channel_ids=channel_ids,
        existing_metadata=channel_metadata
    )
    
    # DEBUG OUTPUT 3: Save enriched metadata
    print(f"\n📦 DEBUG OUTPUT 3: Enriched Metadata")
    debug_enriched = {
        "total_channels_enriched": len(enriched_metadata),
        "channels": [
            {
                "channel_id": ch_id,
                "name": metadata.get("name", "Unknown"),
                "subscribers": metadata.get("subscribers", 0),
                "video_count": metadata.get("video_count", 0),
                "country": metadata.get("country", "unknown"),
                "frequency": channel_frequency.get(ch_id, 0),
                "description_preview": (metadata.get("description", "")[:200] + "...") if len(metadata.get("description", "")) > 200 else metadata.get("description", "")
            }
            for ch_id, metadata in enriched_metadata.items()
        ]
    }
    save_debug_json(debug_enriched, "03_enriched_metadata.json",
                   f"Full metadata for {len(enriched_metadata)} channels")
    
    df_enriched = pd.DataFrame(debug_enriched["channels"])
    save_debug_csv(df_enriched, "03_enriched_metadata.csv",
                  "Enriched metadata in CSV format")
    
    
    # ============================================
    # STEP 4: Filter by Threshold
    # ============================================
    print(f"\n{'='*70}")
    print("PHASE 3: FILTERING BY FREQUENCY THRESHOLD")
    print(f"{'='*70}")
    
    # Calculate threshold (20% of videos = appears in 20% of searches)
    min_frequency = max(2, int(len(titles) * 0.20))  # At least 2, or 20% of titles
    
    candidates = filter_by_frequency_threshold(
        channel_frequency=channel_frequency,
        channel_metadata=enriched_metadata,
        min_frequency=min_frequency,
        total_searches=len(titles)
    )
    
    if not candidates:
        print(f"\n⚠️ No candidates above threshold {min_frequency}!")
        print("   Try lowering threshold or checking seed videos")
        return
    
    # Display results
    print(f"\n🎯 CANDIDATE COMPETITORS:")
    print(f"{'Rank':<6}{'Channel Name':<40}{'Freq':<8}{'Score':<8}{'Subs':<12}")
    print("-" * 80)
    
    for i, cand in enumerate(candidates, 1):
        print(f"{i:<6}{cand['channel_name'][:38]:<40}"
              f"{cand['appearances']:<8}"
              f"{cand['frequency_score']:.2f}   "
              f"{cand['subscribers']:>10,}")
    
    # DEBUG OUTPUT 4: Save filtered candidates
    print(f"\n📦 DEBUG OUTPUT 4: Filtered Candidates")
    debug_candidates = {
        "threshold_used": {
            "min_frequency": min_frequency,
            "total_searches": len(titles),
            "percentage": round((min_frequency / len(titles)) * 100, 1)
        },
        "total_candidates": len(candidates),
        "candidates": candidates
    }
    save_debug_json(debug_candidates, "04_filtered_candidates.json",
                   f"{len(candidates)} candidates above threshold {min_frequency}/{len(titles)}")
    
    
    # ============================================
    # STEP 5: Export Final Results
    # ============================================
    print(f"\n{'='*70}")
    print("PHASE 4: EXPORTING FINAL RESULTS")
    print(f"{'='*70}")
    
    # Convert to DataFrame
    df_candidates = pd.DataFrame(candidates)
    
    # Export to main output CSV
    output_csv = BASE / "competitors_frequency.csv"
    df_candidates.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ SUCCESS!")
    print(f"   Saved {len(candidates)} candidates to '{output_csv.name}'")
    
    # Also save to debug directory
    save_debug_csv(df_candidates, "05_final_output.csv",
                  "Final competitor list (same as competitors_frequency.csv)")
    
    # Summary stats
    print(f"\n📊 SUMMARY:")
    print(f"   Total unique channels found: {len(channel_frequency)}")
    print(f"   Candidates above threshold: {len(candidates)}")
    print(f"   Frequency threshold used: {min_frequency}/{len(titles)} (20%)")
    
    if candidates:
        avg_freq = sum(c['frequency'] for c in candidates) / len(candidates)
        print(f"   Average frequency: {avg_freq:.1f}/{len(titles)}")
        print(f"   Top channel: {candidates[0]['channel_name']} ({candidates[0]['appearances']})")
    
    # DEBUG OUTPUT 5: Save execution summary
    print(f"\n📦 DEBUG OUTPUT 5: Execution Summary")
    summary = {
        "execution_time": datetime.now().isoformat(),
        "seed_channel": {
            "name": seed_channel_name,
            "id": seed_channel_id
        },
        "input": {
            "total_videos": len(titles),
            "video_titles": titles
        },
        "results": {
            "total_unique_channels_found": len(channel_frequency),
            "threshold_used": {
                "min_frequency": min_frequency,
                "total_searches": len(titles),
                "percentage": round((min_frequency / len(titles)) * 100, 1)
            },
            "candidates_above_threshold": len(candidates),
            "average_frequency": round(sum(c['frequency'] for c in candidates) / len(candidates), 1) if candidates else 0
        },
        "top_10_candidates": [
            {
                "rank": i,
                "name": c['channel_name'],
                "frequency": c['frequency'],
                "frequency_score": c['frequency_score'],
                "subscribers": c['subscribers']
            }
            for i, c in enumerate(candidates[:10], 1)
        ]
    }
    save_debug_json(summary, "06_execution_summary.json",
                   "Complete execution summary with all key metrics")
    
    print(f"\n{'='*70}")
    print("DEBUG FILES CREATED:")
    print(f"{'='*70}")
    print(f"All debug files saved to: {DEBUG_DIR}/")
    print(f"Files created:")
    print(f"  1. 01_input_data.json - Input video titles")
    print(f"  2. 02_raw_frequency.json/.csv - Raw frequency counts")
    print(f"  3. 03_enriched_metadata.json/.csv - Full channel metadata")
    print(f"  4. 04_filtered_candidates.json - Candidates after filtering")
    print(f"  5. 05_final_output.csv - Final competitor list")
    print(f"  6. 06_execution_summary.json - Complete run summary")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review competitors_frequency.csv in main directory")
    print("2. Check debug_outputs/ folder for detailed step-by-step data")
    print("3. Verify expected competitors (Thomas Frank, etc.) are present")
    print("4. If results look good, proceed to Phase 2 (LLM validation)")
    print("5. If results need tuning, adjust frequency threshold and re-run")


if __name__ == "__main__":
    main()
