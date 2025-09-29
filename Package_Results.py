import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Process multiple input directories and one output directory.")
    parser.add_argument(
        "input_dirs",
        nargs="+",
        type=str,
        help="List of input directory paths"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    return parser.parse_args()


""" This script aggregates results from multiple experiments stored in different input directories.
It reads 'Model_Output_sentence_classification.csv' and 'Similarity_Scores.csv' from each input directory,
adds an 'experiment_name' column based on the input directory name, and concatenates them into single DataFrames.
The combined DataFrames are then saved as 'combined_results.parquet' and 'combined_similarity_scores.parquet' in the specified output directory.
"""
if __name__ == "__main__":
    args = parse_args()
    print(f"Input directories: {args.input_dirs}")
    print(f"Output directory: {args.output_dir}")

    experiments_dfs = []
    score_dfs = []
    num_samples = {}
    for input_dir in args.input_dirs:
        print(f"Processing input directory: {input_dir}")
        
        experiment_name = os.path.basename(os.path.normpath(input_dir))

        df = pd.read_csv(os.path.join(input_dir, "Model_Output_sentence_classification.csv"), index_col=0)
        df.index = df.index.astype(str)
        df["experiment_name"] = experiment_name
        experiments_dfs.append(df)
        num_samples[experiment_name] = len(df.index.unique())


        if os.path.exists(os.path.join(input_dir, "Similarity_Scores.csv")):
            score_df = pd.read_csv(os.path.join(input_dir, "Similarity_Scores.csv")) 
            score_df["experiment_name"] = experiment_name
            score_dfs.append(score_df)
        else:
            print(f"Warning: Similarity_Scores.csv not found in {input_dir}, skipping.")

    print(num_samples)

    combined_df = pd.concat(experiments_dfs)

    print(f"Saving combined results to {os.path.join(args.output_dir, 'combined_results.parquet')}")
    combined_df.to_parquet(os.path.join(args.output_dir, "combined_results.parquet"), index=True)

    if not score_dfs:
        print("No Similarity_Scores.csv files found in any input directory. Skipping similarity scores aggregation.")
    else:
        combined_score_df = pd.concat(score_dfs, ignore_index=True)
        print(f"Saving combined similarity scores to {os.path.join(args.output_dir, 'combined_similarity_scores.parquet')}")
        combined_score_df.to_parquet(os.path.join(args.output_dir, "combined_similarity_scores.parquet"), index=False)   
    
    assert len(combined_df) == sum(len(df) for df in experiments_dfs), "Row count mismatch after concatenation!"
     