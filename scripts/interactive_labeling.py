import pandas as pd
import os
import sys

# Adjust the path to import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from api_v2.utils.api_predict import BankTransactionCategorizerHF
    from utils.dicts import categories
except ImportError:
    print("Error: Could not import necessary modules.")
    print("Please ensure you are running the script from the root directory of the project.")
    sys.exit(1)

INPUT_FILE = 'data/needs_labelling.csv'
OUTPUT_FILE = 'data/newly_labeled_data.csv'

def get_user_choice(prompt, options):
    """Generic function to get a valid user choice from a numbered list."""
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def interactive_labelling():
    """Main function to run the interactive labelling tool."""
    # --- Load Data ---
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at '{INPUT_FILE}'")
        return

    unlabeled_df = pd.read_csv(INPUT_FILE)
    
    # --- Load labeled data to avoid re-labeling ---
    labeled_data = []
    if os.path.exists(OUTPUT_FILE):
        labeled_df = pd.read_csv(OUTPUT_FILE)
        already_labeled_descriptions = set(labeled_df['Description'])
        unlabeled_df = unlabeled_df[~unlabeled_df['Description'].isin(already_labeled_descriptions)]
        labeled_data = labeled_df.to_dict('records')
        print(f"Loaded {len(labeled_data)} previously labeled records. Skipping them.")

    if unlabeled_df.empty:
        print("All descriptions from the input file have already been labeled. Exiting.")
        return
        
    # --- Initialize model ---
    print("Loading the model for initial predictions...")
    categorizer = BankTransactionCategorizerHF()
    print("Model loaded.")

    category_list = sorted(categories.keys())

    # --- Interactive Loop ---
    for index, row in unlabeled_df.iterrows():
        description = row['Description']
        
        # Get model's prediction
        temp_df = pd.DataFrame([{'Description': description}])
        prediction_df = categorizer.predict(temp_df)
        pred_row = prediction_df.iloc[0]
        
        print("\n" + "="*80)
        print(f"Transaction {index + 1}/{len(unlabeled_df) + len(labeled_data)}: {description}")
        print("-" * 30)
        print(f"Model Prediction -> Category: '{pred_row['Category']}', Subcategory: '{pred_row['Subcategory']}'")
        print(f"Confidence -> Category: {pred_row['CategoryConfidence']:.2f}, Subcategory: {pred_row['SubcategoryConfidence']:.2f}")
        print("-" * 30)

        # --- User Interaction ---
        action = input("(y)es, looks good | (c)hange label | (s)kip | (q)uit and save\n> ").lower()

        if action == 'y':
            final_cat = pred_row['Category']
            final_sub = pred_row['Subcategory']
        elif action == 'c':
            # --- Change Category ---
            print("\n--- Select a new Category ---")
            for i, cat in enumerate(category_list):
                print(f"[{i+1}] {cat}")
            chosen_cat = get_user_choice("> ", category_list)

            # --- Change Subcategory ---
            subcategory_list = sorted(categories[chosen_cat])
            print(f"\n--- Select a new Subcategory for '{chosen_cat}' ---")
            for i, sub in enumerate(subcategory_list):
                print(f"[{i+1}] {sub}")
            chosen_sub = get_user_choice("> ", subcategory_list)
            
            final_cat = chosen_cat
            final_sub = chosen_sub
            print(f"Label changed to: '{final_cat}' -> '{final_sub}'")

        elif action == 's':
            print("Skipping...")
            continue
        elif action == 'q':
            print("Quitting and saving progress...")
            break
        else:
            print("Invalid action. Skipping.")
            continue

        labeled_data.append({'Description': description, 'Category': final_cat, 'Sub_Category': final_sub})

    # --- Save Results ---
    if labeled_data:
        final_df = pd.DataFrame(labeled_data)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(labeled_data)} labeled transactions to '{OUTPUT_FILE}'.")
    else:
        print("\nNo new transactions were labeled.")

if __name__ == "__main__":
    interactive_labelling() 