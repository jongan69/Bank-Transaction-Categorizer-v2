import pandas as pd
import json
import os
import random
import argparse
from datetime import datetime, timedelta

def generate_random_string(pattern):
    """Generates a random string based on a pattern (e.g., '###-##')."""
    s = ""
    for char in pattern:
        if char == '#':
            s += str(random.randint(0, 9))
        elif char == 'X':
            s += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        else:
            s += char
    return s

def generate_random_date(pattern):
    """Generates a random date string based on a format pattern."""
    # A fixed date range for plausible transaction dates
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    random_date = start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds())),
    )
    if pattern == "MM/DD/YYYY":
        return random_date.strftime("%m/%d/%Y")
    elif pattern == "MMDD":
        return random_date.strftime("%m%d")
    elif pattern == "YYYYMMDD":
        return random_date.strftime("%Y%m%d")
    return random_date.strftime("%Y-%m-%d")

def generate_data(category, subcategory, num_samples, templates, output_path):
    """
    Generates synthetic transaction data for a specific category/subcategory.
    """
    try:
        cat_data = templates[category][subcategory]
        template_list = cat_data["templates"]
        placeholders = cat_data["placeholders"]
    except KeyError:
        print(f"Warning: No template found for {category}/{subcategory}. Skipping.")
        return

    generated_records = []
    for _ in range(num_samples):
        template = random.choice(template_list)
        description = template
        
        for key, values in placeholders.items():
            placeholder_key = f"{{{key}}}"
            if placeholder_key in description:
                chosen_value = random.choice(values)
                if '#' in chosen_value or 'X' in chosen_value:
                    replacement = generate_random_string(chosen_value)
                elif 'YYYY' in chosen_value or 'MM' in chosen_value or 'DD' in chosen_value:
                    replacement = generate_random_date(chosen_value)
                else:
                    replacement = chosen_value
                description = description.replace(placeholder_key, replacement, 1)

        generated_records.append({
            'Description': description,
            'Category': category,
            'Sub_Category': subcategory
        })

    if generated_records:
        output_df = pd.DataFrame(generated_records)
        if os.path.exists(output_path):
            output_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            output_df.to_csv(output_path, index=False)
        print(f"  - Generated and saved {len(output_df)} records for {category}/{subcategory}")

def balance_dataset_automatically(target_count, training_data_path, templates_path, output_path):
    """
    Analyzes the training data distribution and generates synthetic data
    for underrepresented categories to meet the target count.
    """
    try:
        with open(templates_path, 'r') as f:
            templates = json.load(f)
        print(f"Successfully loaded templates from {templates_path}")
    except FileNotFoundError:
        print(f"Error: Templates file not found at {templates_path}")
        return

    try:
        df = pd.read_csv(training_data_path)
        df.columns = df.columns.str.strip() # Strip whitespace from column names
        print(f"Loaded {len(df)} records from {training_data_path} for analysis.")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {training_data_path}")
        return

    # Clear the generated data file to start fresh
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Cleared existing generated data file: {output_path}")

    print(f"\nStarting automatic data generation to reach target of {target_count} samples per subcategory...")

    category_counts = df.groupby(['Category', 'Sub_Category']).size()

    for category, subcategories in templates.items():
        for subcategory_name, _ in subcategories.items():
            current_count = category_counts.get((category, subcategory_name), 0)
            
            if current_count < target_count:
                num_to_generate = target_count - current_count
                print(f"Found underrepresented subcategory: {category}/{subcategory_name} (Current: {current_count}, Target: {target_count})")
                generate_data(category, subcategory_name, num_to_generate, templates, output_path)

    print("\nAutomatic data balancing complete.")
    if not os.path.exists(output_path):
         print("No new data was generated as all templated categories meet the target count.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data from templates. Can run in manual or automatic mode.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--auto", action="store_true", help="Run in automatic balancing mode.")
    mode_group.add_argument("--manual", action="store_true", help="Run in manual generation mode.")

    # Arguments for Auto mode
    parser.add_argument("--target-count", type=int, default=1000, help="Target number of samples per subcategory (for auto mode).")
    parser.add_argument("--training-data", default="data/training_data_v2.csv", help="Path to the training data to analyze for balancing (for auto mode).")

    # Arguments for Manual mode
    parser.add_argument("--category", help="The main category to generate data for (for manual mode).")
    parser.add_argument("--subcategory", help="The subcategory to generate data for (for manual mode).")
    parser.add_argument("--num-samples", type=int, help="Number of samples to create (for manual mode).")

    # Common arguments
    parser.add_argument("--templates", default="scripts/synthetic_data_templates.json", help="Path to the templates JSON file.")
    parser.add_argument("--output", default="data/generated_data.csv", help="Path to save the generated CSV data.")

    args = parser.parse_args()

    if args.auto:
        balance_dataset_automatically(args.target_count, args.training_data, args.templates, args.output)
    elif args.manual:
        if not all([args.category, args.subcategory, args.num_samples]):
            parser.error("--manual mode requires --category, --subcategory, and --num-samples.")
        
        try:
            with open(args.templates, 'r') as f:
                templates = json.load(f)
        except FileNotFoundError:
            print(f"Error: Templates file not found at {args.templates}")
            exit()
            
        generate_data(args.category, args.subcategory, args.num_samples, templates, args.output) 