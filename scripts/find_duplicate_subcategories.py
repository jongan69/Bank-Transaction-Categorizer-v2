import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dicts import categories
from collections import defaultdict

# Build a mapping from subcategory to all parent categories
subcategory_to_parents = defaultdict(list)
for parent, subcats in categories.items():
    for subcat in subcats:
        subcategory_to_parents[subcat].append(parent)

# Find duplicates
duplicates = {subcat: parents for subcat, parents in subcategory_to_parents.items() if len(parents) > 1}

if not duplicates:
    print("No duplicate subcategory names found. All subcategories are unique.")
else:
    print("Duplicate subcategory names found (subcategory: [parent categories]):\n")
    for subcat, parents in duplicates.items():
        print(f"  {subcat}: {parents}")
    print(f"\nTotal duplicates: {len(duplicates)}") 