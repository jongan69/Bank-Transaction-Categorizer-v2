import sys
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model import DistilBertForTransactionClassification, DistilBertForTransactionClassificationConfig, train_model
from utils.data_prep import DataPreprocessor

# Parameters
DATA_PATH = 'data/main_combined.csv'
MODEL_SAVE_PATH = 'models/distilbert-transaction-classifier'
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
print(f"Using device: {DEVICE}")

# Data Preparation
data_preprocessor = DataPreprocessor(DATA_PATH)
num_categories, num_subcategories = data_preprocessor.get_cat_sub_numbers()
data_preprocessor.clean_dataframe()
data_preprocessor.tokenize_data(max_len=32)
X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = data_preprocessor.prepare_data()

# Convert labels to tensors
y_cat_train_tensor = torch.tensor(np.array(y_cat_train).argmax(axis=1), dtype=torch.long)
y_cat_test_tensor = torch.tensor(np.array(y_cat_test).argmax(axis=1), dtype=torch.long)
y_sub_train_tensor = torch.tensor(np.array(y_sub_train).argmax(axis=1), dtype=torch.long)
y_sub_test_tensor = torch.tensor(np.array(y_sub_test).argmax(axis=1), dtype=torch.long)

train_input_ids = torch.tensor(X_train)
val_input_ids = torch.tensor(X_test)

# Create datasets and dataloaders
train_dataset = TensorDataset(train_input_ids, y_cat_train_tensor, y_sub_train_tensor)
val_dataset = TensorDataset(val_input_ids, y_cat_test_tensor, y_sub_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print('Train batches:', len(train_dataloader))
print('Validation batches:', len(val_dataloader))

# Initialize model with custom config
config = DistilBertForTransactionClassificationConfig.from_pretrained(
    'distilbert-base-uncased',
    num_categories=num_categories,
    num_subcategories=num_subcategories
)
model = DistilBertForTransactionClassification.from_pretrained('distilbert-base-uncased', config=config)
model.to(DEVICE)

print('Training the unified model...')
train_model(model, train_dataloader, val_dataloader, EPOCHS, LEARNING_RATE, DEVICE)

# Save the model using save_pretrained
model.save_pretrained(MODEL_SAVE_PATH)
print(f'Unified model saved to {MODEL_SAVE_PATH}') 