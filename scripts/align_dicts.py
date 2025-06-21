import json
import os
from pprint import pformat
import pandas as pd

def align_dicts():
    """
    Reads category rules from category_rules.json, filters them based on
    the contents of main_combined.csv, and updates utils/dicts.py.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    rules_path = os.path.join(base_dir, 'scripts', 'category_rules.json')
    dicts_path = os.path.join(base_dir, 'utils', 'dicts.py')
    data_path = os.path.join(base_dir, 'data', 'main_combined.csv')

    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        main_df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found.")
        return
    except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
        print(f"Error reading or parsing a file: {e}")
        return

    data_pairs = set()
    for _, row in main_df.iterrows():
        cat = str(row['Category']).strip()
        sub = str(row.get('Sub_Category', 'nan')).strip()
        if cat != 'nan':
            data_pairs.add((cat, None))
        if sub != 'nan':
            data_pairs.add((cat, sub))

    new_categories = {}
    for category, subcategories in rules.items():
        if isinstance(subcategories, dict):
            # Only include the category if it's in the data
            if (category, None) in data_pairs:
                # Filter subcategories to only include those present in the data
                present_subcategories = [sub for sub in subcategories.keys() if (category, sub) in data_pairs]
                if present_subcategories:
                    new_categories[category] = present_subcategories
    
    dict_content = f"categories = {pformat(new_categories, indent=4, width=120)}\n"

    try:
        with open(dicts_path, 'w') as f:
            f.write(dict_content)
        print(f"Successfully updated and aligned {dicts_path} with categories from {rules_path} and {data_path}")
    except IOError as e:
        print(f"Error writing to {dicts_path}: {e}")

if __name__ == '__main__':
    align_dicts() 