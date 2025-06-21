import pandas as pd
import torch
import os
import json
import re
from torch.utils.data import TensorDataset, DataLoader
from utils.data_prep import BertDataPreparation
from utils.model import DistilBertForTransactionClassification, get_device
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer
from .dicts import categories

def normalize_description(description):
    """Cleans and standardizes transaction descriptions."""
    description = description.lower()
    description = re.sub(r'\d{2}[/-]\d{2}', '', description)
    description = re.sub(r'#\w+', '', description)
    description = re.sub(r'confirmation# \w+', '', description)
    description = re.sub(r'id:\w+', '', description)
    description = re.sub(r'\b[a-z0-9]*travel[a-z0-9]*\b', 'travel', description)
    description = re.sub(r'\b\w{20,}\b', '', description)
    description = re.sub(r'x{4,}', '', description)
    description = re.sub(r'\s+', ' ', description).strip()
    return description

def label_transaction(description, rules):
    """Labels a transaction based on keyword matching."""
    normalized_description = normalize_description(description)
    for category, subcategories in rules.items():
        for subcategory, keywords in subcategories.items():
            for keyword in keywords:
                if re.search(keyword, normalized_description, re.IGNORECASE):
                    return category, subcategory
    return None, None

class BankTransactionCategorizer:
    def __init__(self, model_path='jonngan/distilbert-transaction-classifier', rules_path='scripts/category_rules.json'):
        self.device = get_device()
        self.model = DistilBertForTransactionClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Try to load rules, handle potential file not found in the new structure
        self.rules = {}
        try:
            if os.path.exists(rules_path):
                self.rules = self.load_rules(rules_path)
        except Exception as e:
            print(f"Warning: Could not load category rules from {rules_path}. Error: {e}")

        self.category_keys = list(categories.keys())
        self.subcategory_keys = [item for sublist in categories.values() for item in sublist]
        self.label_encoder_cat = LabelEncoder().fit(self.category_keys)
        self.label_encoder_subcat = LabelEncoder().fit(self.subcategory_keys)

    def load_rules(self, rules_path):
        with open(rules_path, 'r') as f:
            return json.load(f)

    def predict(self, df):
        if self.rules:
            df['Category'], df['Sub_Category'] = zip(*df['Description'].apply(lambda x: label_transaction(x, self.rules)))
        else:
            df['Category'] = None
            df['Sub_Category'] = None


        unclassified_mask = df['Category'].isnull()
        unclassified_df = df[unclassified_mask].copy()

        if not unclassified_df.empty:
            df_obj = BertDataPreparation(unclassified_df, self.tokenizer, self.label_encoder_cat, self.label_encoder_subcat, max_len=32)
            tokenized_data = df_obj.tokenize_data_for_prediction()
            
            input_ids = tokenized_data['input_ids'].to(self.device)
            attention_mask = tokenized_data['attention_mask'].to(self.device)
            
            predict_dataset = TensorDataset(input_ids)
            predict_dataloader = DataLoader(predict_dataset, batch_size=8)
            
            all_cat_preds, all_sub_preds = [], []

            with torch.no_grad():
                for batch in predict_dataloader:
                    input_ids_batch = batch[0].to(self.device)
                    attention_mask_batch = (input_ids_batch != self.tokenizer.pad_token_id).long().to(self.device)
                    cat_logits, sub_logits = self.model(input_ids_batch, attention_mask=attention_mask_batch)
                    
                    cat_preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
                    sub_preds = torch.argmax(sub_logits, dim=1).cpu().numpy()

                    all_cat_preds.extend(cat_preds)
                    all_sub_preds.extend(sub_preds)

            category_names = self.label_encoder_cat.inverse_transform(all_cat_preds)
            subcategory_names = self.label_encoder_subcat.inverse_transform(all_sub_preds)
            
            df.loc[unclassified_mask, 'Category'] = category_names
            df.loc[unclassified_mask, 'Sub_Category'] = subcategory_names

        df.fillna({'Category': 'Unclassified_Miscellaneous', 'Sub_Category': 'Unknown'}, inplace=True)
        return df 