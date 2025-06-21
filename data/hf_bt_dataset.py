from datasets import load_dataset
import csv
import json
import traceback
import re

# Load dataset
ds = load_dataset("tusharshah2006/bank_statements_transactions")

# Extract timestamp and description from each entry
rows = []
# Regex to find a date pattern, allowing for whitespace around separators
date_pattern = re.compile(r'(\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{2,4})')

def clean_text(text):
    """Replaces newlines with spaces and strips leading/trailing whitespace."""
    if isinstance(text, str):
        return text.replace('\n', ' ').replace('\r', ' ').strip()
    return text

try:
    for row in ds['train']:
        # The ground_truth is a JSON string, so we need to parse it
        gt_data = json.loads(row['ground_truth'])
        gt_parse = gt_data.get('gt_parse', {})
        if gt_parse:
            entries = gt_parse.get('bank_stmt_entries')
            if entries:
                for entry in entries:
                    timestamp = entry.get("TXN_DATE")
                    description = entry.get("TXN_DESC")

                    # Clean timestamp and description
                    timestamp = clean_text(timestamp)
                    description = clean_text(description)

                    # Try to extract a clean date from the timestamp field
                    if timestamp:
                        match = date_pattern.search(timestamp)
                        if match:
                            # Remove spaces from the extracted date
                            timestamp = match.group(0).replace(' ', '')
                    
                    if timestamp and description:
                        rows.append({
                            "timestamp": timestamp,
                            "Description": description
                        })
except Exception as e:
    print("An error occurred during processing:")
    print(traceback.format_exc())

# Save to CSV, quoting all fields to be safe
if rows:
    with open("./data/hf_bt_dataset.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["timestamp", "Description"],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(rows)
    print("CSV file 'hf_bt_dataset.csv' written with", len(rows), "rows.")
else:
    print("No rows to write to CSV.")
