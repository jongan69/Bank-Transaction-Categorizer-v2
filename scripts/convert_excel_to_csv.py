import pandas as pd
import argparse
import os

def convert_excel_to_csv(input_path, output_path):
    """
    Reads an Excel file and converts it to a CSV file.

    Args:
        input_path (str): The path to the input Excel file.
        output_path (str): The path where the output CSV file will be saved.
    """
    try:
        print(f"Reading Excel file from: {input_path}")
        # Reading the excel file
        df = pd.read_excel(input_path)

        print(f"Writing CSV file to: {output_path}")
        # Writing the dataframe to a CSV file
        df.to_csv(output_path, index=False)

        print("Conversion completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert an Excel file (.xlsx) to a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Path to the input Excel file. Example: 'data/source.xlsx'"
    )
    parser.add_argument(
        "output_file",
        help="Path for the output CSV file. Example: 'data/destination.csv'"
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_excel_to_csv(args.input_file, args.output_file) 