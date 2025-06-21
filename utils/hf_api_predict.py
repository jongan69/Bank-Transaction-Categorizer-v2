"""
BankTransactionCategorizerHF: Use a model from Hugging Face Hub (e.g., jonngan/distilbert-hypopt-transaction-classifier)

Example usage:
    from utils.hf_api_predict import BankTransactionCategorizerHF
    categorizer = BankTransactionCategorizerHF(
        repo_id='jonngan/distilbert-hypopt-transaction-classifier'
    )
    # df = pd.DataFrame([...])
    # result_df = categorizer.predict(df)
"""
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
from .data_prep import DataPreprocessor
from .dicts import categories
from .model import DistilBertForTransactionClassification, DistilBertForTransactionClassificationConfig

class BankTransactionCategorizerHF:
    def __init__(self, repo_id='jonngan/distilbert-hypopt-transaction-classifier'):
        self.repo_id = repo_id
        self.category_keys = list(categories.keys())
        self.category_values = [item for sublist in categories.values() for item in sublist]
        self.num_categories = len(self.category_keys)
        self.num_subcategories = len(self.category_values)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Use MPS for Apple Silicon if available
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_encoder_cat = LabelEncoder()
        self.label_encoder_subcat = LabelEncoder()
        self.label_encoder_cat.fit(self.category_keys)
        self.label_encoder_subcat.fit(self.category_values)
        self.model = self._load_model(self.repo_id)

    def _load_model(self, repo_id):
        config = DistilBertForTransactionClassificationConfig.from_pretrained(
            repo_id,
            num_categories=self.num_categories,
            num_subcategories=self.num_subcategories
        )
        model = DistilBertForTransactionClassification.from_pretrained(repo_id, config=config)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, df: pd.DataFrame):
        df_obj = DataPreprocessor(df)
        df_obj.clean_dataframe()
        X_predict = df_obj.tokenize_predict_data(max_len=32)
        predict_input_ids = torch.tensor(X_predict, dtype=torch.long)
        attention_mask = (predict_input_ids != 0).long()
        predict_dataset = TensorDataset(predict_input_ids, attention_mask)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        categories, subcategories, descriptions = [], [], []
        for batch in predict_dataloader:
            input_ids, attention_mask = [item.to(self.device) for item in batch]
            with torch.no_grad():
                category_probs, subcategory_probs = self.model(input_ids, attention_mask=attention_mask)
                category_predictions = category_probs.argmax(dim=-1)
                subcategory_predictions = subcategory_probs.argmax(dim=-1)
            for i in range(input_ids.size(0)):
                category_name = self.label_encoder_cat.inverse_transform([category_predictions[i].item()])[0]
                subcategory_name = self.label_encoder_subcat.inverse_transform([subcategory_predictions[i].item()])[0]
                single_input_ids = input_ids[i].to('cpu')
                tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                description = self.tokenizer.convert_tokens_to_string([token for token in tokens if token != "[PAD]"]).strip()
                categories.append(category_name)
                subcategories.append(subcategory_name)
                descriptions.append(description)
        result_df = pd.DataFrame({
            'Description': descriptions,
            'Category': categories,
            'Subcategory': subcategories
        })
        return result_df 