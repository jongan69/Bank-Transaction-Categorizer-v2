import pandas as pd
from collections import Counter

# Load the dataset
try:
    df = pd.read_csv('data/main_combined.csv')
except FileNotFoundError:
    print("Error: The file 'data/main_combined.csv' was not found.")
    exit()

# Ensure the required columns exist
if 'Category' not in df.columns or 'Sub_Category' not in df.columns:
    print("Error: The DataFrame must contain 'Category' and 'Sub_Category' columns.")
    exit()

# --- Category Analysis ---
print("--- Category Distribution ---")
category_counts = Counter(df['Category'])
total_categories = len(df['Category'])

# Sort by count, ascending
sorted_categories = sorted(category_counts.items(), key=lambda item: item[1])

for category, count in sorted_categories:
    percentage = (count / total_categories) * 100
    print(f"{category:<30} | Count: {count:<5} | Percentage: {percentage:.2f}%")

print("\n" + "="*50 + "\n")

# --- Sub_Category Analysis ---
print("--- Sub_Category Distribution ---")
subcategory_counts = Counter(df['Sub_Category'])
total_subcategories = len(df['Sub_Category'])

# Sort by count, ascending, and show the top 30 least common
sorted_subcategories = sorted(subcategory_counts.items(), key=lambda item: item[1])

print("Top 30 least common sub-categories:")
for subcategory, count in sorted_subcategories[:30]:
    percentage = (count / total_subcategories) * 100
    print(f"{subcategory:<30} | Count: {count:<5} | Percentage: {percentage:.2f}%")

print("\n" + "="*50 + "\n")

# --- Category and Sub_Category Combination Analysis ---
print("--- Category/Sub_Category Combination Distribution ---")
# Group by Category and then count Sub_Categories within each group
grouped = df.groupby('Category')['Sub_Category'].value_counts().rename('count').reset_index()

# Sort by category, then by count within each category
sorted_grouped = grouped.sort_values(['Category', 'count'], ascending=[True, True])

current_category = ""
for _, row in sorted_grouped.iterrows():
    if row['Category'] != current_category:
        if current_category != "":
            print("-" * 20)
        current_category = row['Category']
        print(f"\nCategory: {current_category}\n")
    
    print(f"  - {row['Sub_Category']:<28} | Count: {row['count']}") 