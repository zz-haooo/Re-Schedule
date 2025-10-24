"""
Convert per-question computed metrics into rank percentages and add them to the dataset.
Only supports pruning_improvement_topM_sum and parent_selection_improvement_topM_sum metrics.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analyze_tree import KAryTreeAnalyzer

def extract_step_and_question(filename: str) -> Tuple[int, int]:
    """
    Extract training step and question number from filename.
    Example: global_step_10_1_overview.txt -> (10, 1)
    Example: Qwen2.5-Math-7B_1_overview.txt -> (0, 1)
    """
    # Match global_step_number_number_overview.txt format
    pattern = r'global_step_(\d+)_(\d+)_overview\.txt'
    match = re.search(pattern, filename)
    
    if match:
        step = int(match.group(1))
        question = int(match.group(2))
        return step, question
    else:
        # Match number_number_overview.txt format
        pattern2 = r'(\d+)_(\d+)_overview\.txt'
        match2 = re.search(pattern2, filename)
        
        if match2:
            step = int(match2.group(1))
            question = int(match2.group(2))
            return step, question
        else:
            # Match modelname_number_overview.txt format, e.g. Qwen2.5-Math-7B_1_overview.txt
            pattern3 = r'^[A-Za-z0-9\.\-]+_(\d+)_overview\.txt$'
            match3 = re.search(pattern3, filename)
            
            if match3:
                step = 0
                question = int(match3.group(1))
                return step, question
            else:
                raise ValueError(f"Cannot extract step and question number from filename {filename}")

def find_overview_files(directory: str) -> List[str]:
    """Find all overview files in directory"""
    overview_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_overview.txt'):
                overview_files.append(os.path.join(root, file))
    
    return overview_files

def analyze_files_and_get_metrics(file_paths: List[str], metric_type: str = 'pruning_improvement_topM_sum', verbose: bool = False, top_m: int = 5) -> Dict[int, float]:
    """
    Batch analyze files and return mapping from question ID to metric value.
    Return format: {question_id: metric_value}
    
    Args:
        file_paths: List of file paths
        metric_type: Metric type: 'pruning_improvement_topM_sum' | 'parent_selection_improvement_topM_sum'
        verbose: Whether to show detailed output
        top_m: M value for top-M calculation
    """
    results = {}
    
    print(f"Starting batch analysis of {len(file_paths)} files...")
    print(f"Analysis metric type: {metric_type}")
    print(f"Top-M: {top_m}")
    
    for file_path in sorted(file_paths):
        try:
            filename = os.path.basename(file_path)
            step, question = extract_step_and_question(filename)
            
            if verbose:
                print(f"Processing file: {filename} (step={step}, question={question})")
            
            # Use KAryTreeAnalyzer to analyze file
            analyzer = KAryTreeAnalyzer()
            result = analyzer.analyze_file(file_path)
            
            # Store corresponding value based on metric type
            if metric_type == 'pruning_improvement_topM_sum':
                # Calculate iterative pruning accuracy improvement top-M sum
                metric_value = analyzer.calculate_pruning_accuracy_improvement_topM_sum(top_m=top_m)
            elif metric_type == 'parent_selection_improvement_topM_sum':
                # Calculate parent selects best child accuracy improvement top-M sum
                metric_value = analyzer.calculate_parent_child_selection_improvement_topM_sum(top_m=top_m)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}. Supported: pruning_improvement_topM_sum, parent_selection_improvement_topM_sum")
            
            # Store result (if multiple files for same question, take max value)
            if question not in results or metric_value > results[question]:
                results[question] = metric_value
                
            if verbose:
                print(f"  Question ID: {question}, {metric_type}: {metric_value:.6f}")
                
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            continue
    
    print(f"Analysis completed, processed {len(results)} questions")
    return results

def calculate_rank_percentages(metric_data: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate rank percentage (0.0-1.0) for each question.
    0.0 indicates highest rank, 1.0 indicates lowest rank.
    
    Args:
        metric_data: Mapping from question ID to metric value
    
    Returns:
        Mapping from question ID to rank percentage
    """
    if not metric_data:
        return {}
    
    # Sort from high to low (highest gets rank percentage 0, lowest gets 1)
    sorted_items = sorted(metric_data.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate rank percentages
    total_count = len(sorted_items)
    rank_percentages = {}
    
    for rank, (question_id, metric_value) in enumerate(sorted_items):
        # Calculate rank percentage: rank / (total_count - 1)
        # This way 0.0 is first place, 1.0 is last place
        if total_count == 1:
            percentage = 0.0
        else:
            percentage = rank / (total_count - 1)
        
        # Round to 2 decimals
        rank_percentages[question_id] = round(percentage, 2)
    
    return rank_percentages

def add_rank_percentages_to_dataset(dataset_path: str, rank_percentages: Dict[int, float], output_path: str) -> bool:
    """
    Add rank percentage data to the original dataset.
    
    Args:
        dataset_path: Original dataset path
        rank_percentages: Mapping from question ID to rank percentage
        output_path: Output dataset path
    
    Returns:
        Whether successful
    """
    try:
        print(f"Reading original dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path)
        
        print(f"Original dataset size: {len(df)} rows")
        print(f"Questions with rank data: {len(rank_percentages)}")
        
        # Create new rank percentage column
        rank_column_name = "metric"
        df[rank_column_name] = np.nan  # Initialize as NaN
        
        # Fill rank percentage values
        filled_count = 0
        for question_id, rank_percentage in rank_percentages.items():
            if 0 <= question_id < len(df):
                df.loc[question_id, rank_column_name] = rank_percentage
                filled_count += 1
        
        print(f"Successfully filled rank percentages for {filled_count} questions")
        print(f"Fill rate: {filled_count}/{len(df)} ({filled_count/len(df)*100:.2f}%)")
        
        # Assign 0.5 (medium rank) to null values
        null_count = df[rank_column_name].isna().sum()
        if null_count > 0:
            print(f"Assigning {null_count} null values to 0.5 (medium rank)")
            df[rank_column_name] = df[rank_column_name].fillna(0.5)
        
        # Calculate rank percentage statistics
        valid_ranks = df[rank_column_name].dropna()
        if len(valid_ranks) > 0:
            print(f"\nRank percentage statistics:")
            print(f"  Valid rank count: {len(valid_ranks)}")
            print(f"  Mean: {valid_ranks.mean():.4f}")
            print(f"  Median: {valid_ranks.median():.4f}")
            print(f"  Std dev: {valid_ranks.std():.4f}")
            print(f"  Min: {valid_ranks.min():.2f}")
            print(f"  Max: {valid_ranks.max():.2f}")
            
            # Rank distribution statistics
            print(f"\nRank distribution:")
            print(f"  Top 10% (0.0-0.1): {len(valid_ranks[valid_ranks <= 0.1])} questions")
            print(f"  Top 25% (0.0-0.25): {len(valid_ranks[valid_ranks <= 0.25])} questions")
            print(f"  Top 50% (0.0-0.5): {len(valid_ranks[valid_ranks <= 0.5])} questions")
            print(f"  Bottom 25% (0.75-1.0): {len(valid_ranks[valid_ranks >= 0.75])} questions")
            print(f"  Bottom 10% (0.9-1.0): {len(valid_ranks[valid_ranks >= 0.9])} questions")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save modified dataset
        print(f"\nSaving modified dataset: {output_path}")
        df.to_parquet(output_path, index=False)
        
        print(f"Dataset modification completed, saved {len(df)} rows")
        print(f"New column: {rank_column_name}")
        
        return True
        
    except Exception as e:
        print(f"Error modifying dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert per-question computed metrics into rank percentages and add them to dataset. Only supports pruning_improvement_topM_sum and parent_selection_improvement_topM_sum.')
    parser.add_argument('directory', help='Directory path containing overview files')
    parser.add_argument('--dataset-path', required=True, help='Original dataset path')
    parser.add_argument('--output-path', required=True, help='Output dataset path')
    parser.add_argument('--metric', choices=['pruning_improvement_topM_sum', 'parent_selection_improvement_topM_sum'], default='parent_selection_improvement_topM_sum',
                       help='Metric type: pruning_improvement_topM_sum (iterative pruning M times accuracy improvement sum) | parent_selection_improvement_topM_sum (parent selects best child M times accuracy improvement sum)')
    parser.add_argument('--topM', type=int, default=5, help='M value for top-M calculation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Directory does not exist: {args.directory}")
        return
    
    # Check if dataset file exists
    if not os.path.exists(args.dataset_path):
        print(f"Dataset file does not exist: {args.dataset_path}")
        return
    
    # Find overview files
    overview_files = find_overview_files(args.directory)
    
    if not overview_files:
        print(f"No overview files found in directory {args.directory}")
        return
    
    print(f"Found {len(overview_files)} overview files")
    
    # Analyze files and get metric data
    metric_data = analyze_files_and_get_metrics(overview_files, args.metric, verbose=args.verbose, top_m=args.topM)
    
    if not metric_data:
        print("Failed to analyze any files successfully")
        return
    
    # Calculate rank percentages
    print(f"\nCalculating rank percentages...")
    rank_percentages = calculate_rank_percentages(metric_data)
    
    print(f"Rank percentage calculation completed for {len(rank_percentages)} questions")
    print(f"Rank range: {min(rank_percentages.values()):.2f} - {max(rank_percentages.values()):.2f}")
    
    # Add rank percentages to dataset
    success = add_rank_percentages_to_dataset(args.dataset_path, rank_percentages, args.output_path)
    
    if not success:
        print("Dataset modification failed")
        return
    
    print(f"\nTask completed!")
    print(f"Modified dataset: {args.output_path}")

if __name__ == "__main__":
    main()
