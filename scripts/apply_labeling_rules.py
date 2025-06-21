import pandas as pd
import json
import os
import re
import argparse

def apply_labelling_rules(rules_path, input_path, output_labeled_path, output_unlabeled_path):
    """
    Applies a set of rules to an unlabeled dataset, splitting it into
    labeled and unlabeled outputs.

    Args:
        rules_path (str): Path to the JSON file with the labeling rules.
        input_path (str): Path to the input CSV of unlabeled data.
        output_labeled_path (str): Path to save the auto-labeled data.
        output_unlabeled_path (str): Path to save the data that couldn't be labeled.
    """
    # --- Load Rules ---
    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        print(f"Successfully loaded rules from {rules_path}")
    except FileNotFoundError:
        print(f"Error: Rules file not found at {rules_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {rules_path}. Please check its format.")
        return

    # --- Load Unlabeled Data ---
    try:
        unlabeled_df = pd.read_csv(input_path)
        print(f"Loaded {len(unlabeled_df)} records from {input_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return

    # --- Prepare for Labeling ---
    labeled_records = []
    unlabeled_indices = []
    
    # Create a flattened list of rules for easier processing
    flat_rules = []
    for category, subcategories in rules.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                # Compile regex for case-insensitive matching
                try:
                    # The \\b is for whole word matching, important for many keywords.
                    # It ensures 'atm' doesn't match 'html'.
                    regex = re.compile(r'\b' + keyword + r'\b', re.IGNORECASE)
                    flat_rules.append({
                        "regex": regex,
                        "category": category,
                        "subcategory": subcategory
                    })
                except re.error as e:
                    print(f"Warning: Could not compile regex for keyword '{keyword}'. Error: {e}")


    # --- Apply Rules ---
    for index, row in unlabeled_df.iterrows():
        description = row['Description']
        matched = False
        for rule in flat_rules:
            if rule['regex'].search(description):
                labeled_records.append({
                    'Description': description,
                    'Category': rule['category'],
                    'Sub_Category': rule['subcategory']
                })
                matched = True
                break  # Stop after the first match
        
        if not matched:
            unlabeled_indices.append(index)

    # --- Create Output DataFrames ---
    auto_labeled_df = pd.DataFrame(labeled_records)
    still_unlabeled_df = unlabeled_df.loc[unlabeled_indices]

    # --- Save Outputs ---
    auto_labeled_df.to_csv(output_labeled_path, index=False)
    still_unlabeled_df.to_csv(output_unlabeled_path, index=False)

    # --- Print Summary ---
    print("\n--- Labeling Summary ---")
    print(f"Total records processed: {len(unlabeled_df)}")
    print(f"Automatically labeled:   {len(auto_labeled_df)}")
    print(f"Still requires labeling: {len(still_unlabeled_df)}")
    print("-" * 25)
    print(f"Labeled data saved to:      '{output_labeled_path}'")
    print(f"Remaining unlabeled data saved to: '{output_unlabeled_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically label transaction data based on a JSON rules file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--rules",
        default="scripts/category_rules.json",
        help="Path to the category rules JSON file."
    )
    parser.add_argument(
        "--input",
        default="data/needs_labelling.csv",
        help="Path to the input CSV file that needs labeling."
    )
    parser.add_argument(
        "--output-labeled",
        default="data/auto_labeled_data.csv",
        help="Path to save the automatically labeled data."
    )
    parser.add_argument(
        "--output-unlabeled",
        default="data/still_needs_labelling.csv",
        help="Path to save the data that still requires manual labeling."
    )

    args = parser.parse_args()
    apply_labelling_rules(args.rules, args.input, args.output_labeled, args.output_unlabeled) 