import pandas as pd
import argparse
import os

def prepare_data_for_labelling(input_path, output_path):
    """
    Reads a raw transaction CSV, extracts, renames, and deduplicates the 
    description column to prepare it for labelling.

    Args:
        input_path (str): The path to the input CSV file (e.g., 'data/bank.csv').
        output_path (str): The path for the output CSV file (e.g., 'data/needs_labelling.csv').
    """
    try:
        print(f"Reading raw data from: {input_path}")
        df = pd.read_csv(input_path)

        # Check if the required column exists
        if 'TRANSACTION DETAILS' not in df.columns:
            print(f"Error: The required column 'TRANSACTION DETAILS' was not found in {input_path}.")
            return

        # Extract, rename, and handle missing values
        descriptions = df[['TRANSACTION DETAILS']].rename(columns={'TRANSACTION DETAILS': 'Description'})
        descriptions.dropna(subset=['Description'], inplace=True)
        descriptions['Description'] = descriptions['Description'].astype(str)
        
        # Remove duplicates
        print(f"Original number of transactions: {len(descriptions)}")
        descriptions.drop_duplicates(subset=['Description'], inplace=True)
        print(f"Number of unique transactions after deduplication: {len(descriptions)}")

        # Save to the output file
        print(f"Saving prepared data to: {output_path}")
        descriptions.to_csv(output_path, index=False)

        print("Data preparation completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare raw transaction data for labelling.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input CSV file. Default: 'data/bank.csv'",
        nargs='?',
        default="data/bank.csv"
    )
    parser.add_argument(
        "output_file",
        help="Path for the output CSV file. Default: 'data/needs_labelling.csv'",
        nargs='?',
        default="data/needs_labelling.csv"
    )
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prepare_data_for_labelling(args.input_file, args.output_file) 