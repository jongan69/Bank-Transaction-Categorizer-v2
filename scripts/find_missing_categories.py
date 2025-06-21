import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

from utils.dicts import categories

DATA_PATH = 'data/main_combined.csv'

def flatten_dict(d):
    pairs = set()
    for cat, subs in d.items():
        pairs.add((cat, None))
        for sub in subs:
            pairs.add((cat, sub))
    return pairs

# 1. Get all categories/subcategories from dict
all_pairs = flatten_dict(categories)

# 2. Get all categories/subcategories from data
main_df = pd.read_csv(DATA_PATH)
data_pairs = set()
for _, row in main_df.iterrows():
    cat = str(row['Category']).strip()
    sub = str(row['Sub_Category']).strip() if 'Sub_Category' in row else None
    data_pairs.add((cat, None))
    if sub and sub != 'nan':
        data_pairs.add((cat, sub))

# 3. Find which pairs are in dict but not in data
missing = all_pairs - data_pairs
if missing:
    print('Categories/subcategories in dict but NOT in main.csv:')
    for cat, sub in sorted(missing, key=lambda x: (x[0], x[1] or "")):
        if sub:
            print(f'  {cat} -> {sub}')
        else:
            print(f'  {cat}')
else:
    print('All categories and subcategories in dict are present in main.csv!') 