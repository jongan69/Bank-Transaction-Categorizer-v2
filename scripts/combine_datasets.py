import pandas as pd
import os
import argparse

def combine_and_deduplicate(main_path, new_labeled_paths, output_path):
    """
    Combines multiple labeled datasets, deduplicates them, and saves the result.

    Args:
        main_path (str): Path to the original main dataset.
        new_labeled_paths (list): A list of paths to new labeled datasets.
        output_path (str): Path to save the final combined dataset.
    """
    # --- Load Original Data ---
    try:
        main_df = pd.read_csv(main_path)
        print(f"Loaded {len(main_df)} records from original dataset: {main_path}")
    except FileNotFoundError:
        print(f"Warning: Main dataset not found at {main_path}. Starting with an empty DataFrame.")
        main_df = pd.DataFrame()

    # --- Load New Data ---
    all_dfs = [main_df]
    for path in new_labeled_paths:
        if os.path.exists(path):
            try:
                new_df = pd.read_csv(path)
                all_dfs.append(new_df)
                print(f"Loaded {len(new_df)} records from new dataset: {path}")
            except Exception as e:
                print(f"Warning: Could not load {path}. Error: {e}")
        else:
            print(f"Info: Dataset not found at {path}. Skipping.")
    
    # --- Combine and Deduplicate ---
    if len(all_dfs) > 1:
        # Concatenate all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal records before deduplication: {len(combined_df)}")

        # Deduplicate based on 'Description', keeping the last entry
        # This prioritizes newly labeled data over the original data
        combined_df.drop_duplicates(subset=['Description'], keep='last', inplace=True)
        
        print(f"Total records after deduplication: {len(combined_df)}")
    else:
        combined_df = main_df
        print("No new datasets to combine. Proceeding with the original data.")


    # --- Save Final Dataset ---
    combined_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully combined datasets and saved to '{output_path}'")
    print("This file is now ready to be used for retraining the model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine original and new labeled datasets for retraining.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--main-data",
        default="data/training_data_v3.csv",
        help="Path to the main original dataset."
    )
    parser.add_argument(
        "--new-data",
        nargs='+',  # Allows multiple new data files
        default=["data/generated_data.csv"],
        help="List of paths to the new labeled data CSV files."
    )
    parser.add_argument(
        "--output",
        default="data/training_data_v4.csv",
        help="Path to save the final combined and deduplicated dataset."
    )

    args = parser.parse_args()
    combine_and_deduplicate(args.main_data, args.new_data, args.output) 