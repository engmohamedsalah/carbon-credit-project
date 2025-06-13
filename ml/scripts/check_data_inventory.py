#!/usr/bin/env python3
"""
Data Inventory Script for ConvLSTM Training
Checks available Sentinel-1, Sentinel-2, and change label data
"""

import os
import glob
import re
from datetime import datetime
from pathlib import Path
import pandas as pd

def extract_date_from_filename(filename):
    """Extract date from various satellite data filename formats"""
    patterns = [
        r'(\d{8})T\d{6}',  # YYYYMMDDTHHMMSS format
        r'(\d{8})',        # YYYYMMDD format
        r'(\d{4})-(\d{2})-(\d{2})'  # YYYY-MM-DD format
    ]
    
    basename = os.path.basename(filename)
    for pattern in patterns:
        match = re.search(pattern, basename)
        if match:
            if len(match.groups()) == 1:
                date_str = match.group(1)
                if len(date_str) == 8:
                    try:
                        return datetime.strptime(date_str, '%Y%m%d')
                    except:
                        continue
            else:
                try:
                    year, month, day = match.groups()
                    return datetime(int(year), int(month), int(day))
                except:
                    continue
    return None

def inventory_sentinel1_data(s1_dir):
    """Inventory Sentinel-1 data"""
    print(f"\n🛰️  SENTINEL-1 DATA INVENTORY")
    print(f"Directory: {s1_dir}")
    
    s1_inventory = {}
    
    # Check for ZIP files
    zip_files = glob.glob(os.path.join(s1_dir, "*.zip"))
    safe_dirs = [d for d in glob.glob(os.path.join(s1_dir, "*.SAFE")) if os.path.isdir(d)]
    
    all_files = zip_files + safe_dirs
    
    if not all_files:
        print("❌ No Sentinel-1 data found")
        return s1_inventory
    
    print(f"📦 Found {len(zip_files)} ZIP files")
    print(f"📁 Found {len(safe_dirs)} SAFE directories")
    
    for file_path in all_files:
        date = extract_date_from_filename(file_path)
        if date:
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
            s1_inventory[date] = {
                'path': file_path,
                'type': 'ZIP' if file_path.endswith('.zip') else 'SAFE',
                'size': file_size
            }
    
    # Sort by date and display
    sorted_dates = sorted(s1_inventory.keys())
    print(f"📅 Date range: {sorted_dates[0].strftime('%Y-%m-%d')} to {sorted_dates[-1].strftime('%Y-%m-%d')}")
    print(f"🗂️  Total scenes: {len(s1_inventory)}")
    
    # Group by year-month
    monthly_counts = {}
    for date in sorted_dates:
        year_month = date.strftime('%Y-%m')
        monthly_counts[year_month] = monthly_counts.get(year_month, 0) + 1
    
    print(f"📊 Monthly distribution:")
    for year_month, count in sorted(monthly_counts.items()):
        print(f"   {year_month}: {count} scenes")
    
    return s1_inventory

def inventory_sentinel2_data(s2_dir):
    """Inventory Sentinel-2 data"""
    print(f"\n🛰️  SENTINEL-2 DATA INVENTORY")
    print(f"Directory: {s2_dir}")
    
    s2_inventory = {}
    
    # Check for ZIP files and SAFE directories
    zip_files = glob.glob(os.path.join(s2_dir, "*.zip"))
    safe_dirs = [d for d in glob.glob(os.path.join(s2_dir, "*.SAFE")) if os.path.isdir(d)]
    
    all_files = zip_files + safe_dirs
    
    if not all_files:
        print("❌ No Sentinel-2 data found")
        return s2_inventory
    
    print(f"📦 Found {len(zip_files)} ZIP files")
    print(f"📁 Found {len(safe_dirs)} SAFE directories")
    
    for file_path in all_files:
        date = extract_date_from_filename(file_path)
        if date:
            file_size = os.path.getsize(file_path) if os.path.isfile(file_path) else "DIR"
            s2_inventory[date] = {
                'path': file_path,
                'type': 'ZIP' if file_path.endswith('.zip') else 'SAFE',
                'size': file_size
            }
    
    # Sort by date and display
    sorted_dates = sorted(s2_inventory.keys())
    print(f"📅 Date range: {sorted_dates[0].strftime('%Y-%m-%d')} to {sorted_dates[-1].strftime('%Y-%m-%d')}")
    print(f"🗂️  Total scenes: {len(s2_inventory)}")
    
    # Group by year-month
    monthly_counts = {}
    for date in sorted_dates:
        year_month = date.strftime('%Y-%m')
        monthly_counts[year_month] = monthly_counts.get(year_month, 0) + 1
    
    print(f"📊 Monthly distribution:")
    for year_month, count in sorted(monthly_counts.items()):
        print(f"   {year_month}: {count} scenes")
    
    return s2_inventory

def inventory_processed_data(prepared_dir):
    """Inventory processed/prepared data"""
    print(f"\n📋 PROCESSED DATA INVENTORY")
    print(f"Directory: {prepared_dir}")
    
    s2_stacks_dir = os.path.join(prepared_dir, "s2_stacks")
    s1_stacks_dir = os.path.join(prepared_dir, "s1_stacks")
    
    # Sentinel-2 stacks
    s2_stacks = glob.glob(os.path.join(s2_stacks_dir, "*_stack.tif")) if os.path.exists(s2_stacks_dir) else []
    print(f"🔄 Processed S2 stacks: {len(s2_stacks)}")
    
    # Sentinel-1 stacks
    s1_stacks = glob.glob(os.path.join(s1_stacks_dir, "*_stack.tif")) if os.path.exists(s1_stacks_dir) else []
    print(f"🔄 Processed S1 stacks: {len(s1_stacks)}")
    
    return s2_stacks, s1_stacks

def inventory_change_labels(change_labels_dir):
    """Inventory change detection labels"""
    print(f"\n🏷️  CHANGE LABELS INVENTORY")
    print(f"Directory: {change_labels_dir}")
    
    if not os.path.exists(change_labels_dir):
        print("❌ Change labels directory not found")
        return {}
    
    change_files = glob.glob(os.path.join(change_labels_dir, "*_change.tif"))
    print(f"🏷️  Found {len(change_files)} change label files")
    
    change_inventory = {}
    for file_path in change_files:
        # Extract date range from filename
        pattern = r'(\d{8})_(\d{8})_change'
        match = re.search(pattern, os.path.basename(file_path))
        if match:
            start_str, end_str = match.groups()
            start_date = datetime.strptime(start_str, '%Y%m%d')
            end_date = datetime.strptime(end_str, '%Y%m%d')
            change_inventory[end_date] = {
                'path': file_path,
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days
            }
    
    if change_inventory:
        sorted_dates = sorted(change_inventory.keys())
        print(f"📅 Change labels date range: {sorted_dates[0].strftime('%Y-%m-%d')} to {sorted_dates[-1].strftime('%Y-%m-%d')}")
    
    return change_inventory

def analyze_temporal_coverage(s1_inventory, s2_inventory, change_inventory):
    """Analyze temporal coverage for time series creation"""
    print(f"\n⏰ TEMPORAL COVERAGE ANALYSIS")
    
    # Get all unique dates
    all_s1_dates = set(s1_inventory.keys())
    all_s2_dates = set(s2_inventory.keys())
    all_change_dates = set(change_inventory.keys())
    
    # Find overlapping periods
    common_dates = all_s1_dates & all_s2_dates
    print(f"📅 Common S1+S2 dates: {len(common_dates)}")
    
    if common_dates:
        sorted_common = sorted(common_dates)
        print(f"   Range: {sorted_common[0].strftime('%Y-%m-%d')} to {sorted_common[-1].strftime('%Y-%m-%d')}")
        
        # Analyze sequence potential
        potential_sequences = analyze_sequence_potential(sorted_common)
        print(f"🔗 Potential 6-month sequences: {potential_sequences['6_month']}")
        print(f"🔗 Potential 3-month sequences: {potential_sequences['3_month']}")
    
    print(f"🏷️  Change labels available: {len(all_change_dates)}")
    
    return {
        's1_dates': len(all_s1_dates),
        's2_dates': len(all_s2_dates),
        'common_dates': len(common_dates),
        'change_dates': len(all_change_dates)
    }

def analyze_sequence_potential(sorted_dates, min_gap_days=30):
    """Analyze potential for creating temporal sequences"""
    sequences = {'3_month': 0, '6_month': 0}
    
    for i in range(len(sorted_dates)):
        # Try to build sequences of different lengths
        for seq_length, seq_name in [(3, '3_month'), (6, '6_month')]:
            sequence_dates = [sorted_dates[i]]
            
            for j in range(1, seq_length):
                # Find next date with minimum gap
                next_candidates = [d for d in sorted_dates[i+j:] 
                                 if (d - sequence_dates[-1]).days >= min_gap_days]
                if next_candidates:
                    sequence_dates.append(next_candidates[0])
                else:
                    break
            
            if len(sequence_dates) == seq_length:
                sequences[seq_name] += 1
    
    return sequences

def create_data_summary_csv(s1_inventory, s2_inventory, change_inventory, output_path):
    """Create a CSV summary of all available data"""
    data_rows = []
    
    # Combine all dates
    all_dates = set()
    all_dates.update(s1_inventory.keys())
    all_dates.update(s2_inventory.keys())
    all_dates.update(change_inventory.keys())
    
    for date in sorted(all_dates):
        row = {
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'day': date.day,
            's1_available': date in s1_inventory,
            's2_available': date in s2_inventory,
            'change_label_available': date in change_inventory,
            's1_path': s1_inventory.get(date, {}).get('path', ''),
            's2_path': s2_inventory.get(date, {}).get('path', ''),
            'change_label_path': change_inventory.get(date, {}).get('path', '')
        }
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(output_path, index=False)
    print(f"📄 Data summary saved to: {output_path}")
    
    return df

def main():
    """Main inventory function"""
    print("🔍 SATELLITE DATA INVENTORY FOR CONVLSTM TRAINING")
    print("=" * 60)
    
    # Define directories
    base_dir = "ml/data"
    s1_dir = os.path.join(base_dir, "sentinel1_downloads")
    s2_dir = os.path.join(base_dir, "sentinel2_downloads")
    change_labels_dir = os.path.join(base_dir, "change_labels")
    prepared_dir = os.path.join(base_dir, "prepared")
    
    # Run inventory
    s1_inventory = inventory_sentinel1_data(s1_dir)
    s2_inventory = inventory_sentinel2_data(s2_dir)
    change_inventory = inventory_change_labels(change_labels_dir)
    
    # Check processed data
    inventory_processed_data(prepared_dir)
    
    # Analyze temporal coverage
    coverage_stats = analyze_temporal_coverage(s1_inventory, s2_inventory, change_inventory)
    
    # Create summary CSV
    summary_path = os.path.join(base_dir, "convlstm_data_inventory.csv")
    create_data_summary_csv(s1_inventory, s2_inventory, change_inventory, summary_path)
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR CONVLSTM TRAINING:")
    print(f"=" * 60)
    
    if coverage_stats['common_dates'] >= 10:
        print("✅ Sufficient temporal data available for ConvLSTM training")
        print("📝 Next steps:")
        print("   1. Process raw data into aligned stacks")
        print("   2. Create temporal sequences dataset")
        print("   3. Begin ConvLSTM training")
    else:
        print("⚠️  Limited temporal overlap between S1 and S2 data")
        print("📝 Recommendations:")
        print("   1. Download additional data to fill gaps")
        print("   2. Consider training with S2-only first")
        print("   3. Add S1 data incrementally")

if __name__ == "__main__":
    main() 