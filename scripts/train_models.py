import sys
import os
import torch
import numpy as np
import optuna
from torch.utils.data import TensorDataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model import DistilBertForTransactionClassification, DistilBertForTransactionClassificationConfig
from utils.data_prep import DataPreprocessor

# Parameters
DATA_PATH = 'data/main_combined.csv'
MODEL_SAVE_PATH = 'models/distilbert-hypopt-transaction-classifier'
TEMP_MODEL_DIR = 'models/temp_best_model'
N_TRIALS = 2  # Number of Optuna trials, you can increase this
PATIENCE = 3 # Early stopping patience

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

train_dataset = TensorDataset(train_input_ids, y_cat_train_tensor, y_sub_train_tensor)
val_dataset = TensorDataset(val_input_ids, y_cat_test_tensor, y_sub_test_tensor)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        model.save_pretrained(self.path)
        self.val_loss_min = val_loss

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 3, 6)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    config = DistilBertForTransactionClassificationConfig.from_pretrained(
        'distilbert-base-uncased',
        num_categories=num_categories,
        num_subcategories=num_subcategories
    )
    model = DistilBertForTransactionClassification.from_pretrained('distilbert-base-uncased', config=config)
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    category_loss_fn = torch.nn.CrossEntropyLoss()
    subcategory_loss_fn = torch.nn.CrossEntropyLoss()
    
    trial_temp_model_dir = f"{TEMP_MODEL_DIR}_trial_{trial.number}"
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=trial_temp_model_dir)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            input_ids, y_cat, y_sub = [item.to(DEVICE) for item in batch]
            optimizer.zero_grad()
            cat_logits, sub_logits = model(input_ids, attention_mask=(input_ids != 0))
            cat_loss = category_loss_fn(cat_logits, y_cat)
            sub_loss = subcategory_loss_fn(sub_logits, y_sub)
            loss = cat_loss + sub_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, y_cat, y_sub = [item.to(DEVICE) for item in batch]
                cat_logits, sub_logits = model(input_ids, attention_mask=(input_ids != 0))
                cat_loss = category_loss_fn(cat_logits, y_cat)
                sub_loss = subcategory_loss_fn(sub_logits, y_sub)
                loss = cat_loss + sub_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Trial {trial.number}, Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return early_stopping.val_loss_min

if __name__ == '__main__':
    if os.path.exists(TEMP_MODEL_DIR):
        shutil.rmtree(TEMP_MODEL_DIR)

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Load the best model from the best trial and save it to the final destination
    best_trial_temp_dir = f"{TEMP_MODEL_DIR}_trial_{study.best_trial.number}"
    print(f"Loading best model from {best_trial_temp_dir}")
    
    config = DistilBertForTransactionClassificationConfig.from_pretrained(best_trial_temp_dir)
    model = DistilBertForTransactionClassification.from_pretrained(best_trial_temp_dir, config=config)
    
    if os.path.exists(MODEL_SAVE_PATH):
        shutil.rmtree(MODEL_SAVE_PATH)
    
    model.save_pretrained(MODEL_SAVE_PATH)
    print(f'Best model saved to {MODEL_SAVE_PATH}')

    # Clean up temporary model directories
    for i in range(N_TRIALS):
        temp_dir = f"{TEMP_MODEL_DIR}_trial_{i}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir) 