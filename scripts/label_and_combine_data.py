import pandas as pd
import json
import re
import os

# Adjust the import path to locate the utils module correctly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dicts import categories

def load_rules(rules_path):
    """
    Loads categorization rules from a JSON file.
    """
    with open(rules_path, 'r') as f:
        return json.load(f)

def normalize_description(description):
    """
    Cleans and standardizes transaction descriptions.
    """
    # Convert to lowercase
    description = str(description).lower()
    # Remove dates (e.g., 06/17, 05-29)
    description = re.sub(r'\d{2}[/-]\d{2}', '', description)
    # Remove confirmation numbers and other long alphanumeric sequences
    description = re.sub(r'#\w+', '', description)
    description = re.sub(r'confirmation# \w+', '', description)
    description = re.sub(r'id:\w+', '', description)
    # Remove long, potentially random alphanumeric strings, but preserve known keywords
    description = re.sub(r'\b[a-z0-9]*travel[a-z0-9]*\b', 'travel', description)
    description = re.sub(r'\b\w{20,}\b', '', description) # Remove long tokens
    description = re.sub(r'x{4,}', '', description) # Remove sequences of 'x'
    # Remove extra whitespace
    description = re.sub(r'\s+', ' ', description).strip()
    return description

def label_transaction(description, rules):
    """
    Labels a transaction based on keyword matching.
    """
    normalized_description = normalize_description(description)
    for category, subcategories in rules.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                if re.search(keyword, normalized_description, re.IGNORECASE):
                    return category, subcategory, True
    return 'Unclassified_Miscellaneous', 'Unknown', False

def main():
    """
    Main function to load, label, combine, and save the data.
    """
    # Define base path to ensure correct file access from script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Load categorization rules
    rules_path = os.path.join(base_dir, 'scripts', 'category_rules.json')
    rules = load_rules(rules_path)

    # Load new transactions
    new_transactions = []
    # Useful for combining with the HF dataset
    # transactions_file_path = os.path.join(base_dir, 'data', 'hf_bt_dataset.csv')
    transactions_file_path = os.path.join(base_dir, 'data', 'transaction_requests.csv')
   
    
    if os.path.exists(transactions_file_path):
        df_new_transactions = pd.read_csv(transactions_file_path)
        new_transactions = df_new_transactions['Description'].tolist()
    else:
        print(f"'{transactions_file_path}' not found. No new transactions will be processed.")

    # Label new transactions and identify those needing review
    labeled_data = []
    needs_review = []
    for desc in new_transactions:
        category, subcategory, classified = label_transaction(desc, rules)
        labeled_data.append({'Description': desc, 'Category': category, 'Sub_Category': subcategory})
        if not classified:
            needs_review.append({'Description': desc, 'Category': '', 'Sub_Category': ''})

    df_new = pd.DataFrame(labeled_data)
    df_review = pd.DataFrame(needs_review)

    # Load existing data
    df_main = pd.read_csv(os.path.join(base_dir, 'data', 'main.csv'))
    
    # Ensure df_main has only the required columns
    required_columns = ['Description', 'Category', 'Sub_Category']
    df_main = df_main[required_columns]

    # Combine dataframes
    df_combined = pd.concat([df_main, df_new], ignore_index=True)
    
    # Remove duplicates
    df_combined.drop_duplicates(subset=['Description'], inplace=True, keep='last')

    # Save the combined dataframe
    output_path = os.path.join(base_dir, 'data', 'main_combined.csv')
    df_combined.to_csv(output_path, index=False)
    print(f"Combined and labeled data saved to {output_path}")

    # Save transactions needing review
    if not df_review.empty:
        review_path = os.path.join(base_dir, 'data', 'needs_review.csv')
        df_review.drop_duplicates(subset=['Description'], inplace=True)
        df_review.to_csv(review_path, index=False)
        print(f"Found {len(df_review)} transactions needing review. Saved to {review_path}")

if __name__ == '__main__':
    main() 