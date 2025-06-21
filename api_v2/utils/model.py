import torch
import torch.nn as nn
from transformers import DistilBertPreTrainedModel, DistilBertModel, DistilBertConfig
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DistilBertForTransactionClassificationConfig(DistilBertConfig):
    model_type = "distilbert_for_transaction_classification"

    def __init__(self, num_categories=20, num_subcategories=70, **kwargs):
        super().__init__(**kwargs)
        self.num_categories = num_categories
        self.num_subcategories = num_subcategories

class DistilBertForTransactionClassification(DistilBertPreTrainedModel):
    config_class = DistilBertForTransactionClassificationConfig

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.category_classifier = nn.Linear(config.dim, config.num_categories)
        self.subcategory_classifier = nn.Linear(config.dim, config.num_subcategories)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        
        category_logits = self.category_classifier(pooled_output)
        subcategory_logits = self.subcategory_classifier(pooled_output)
        
        return category_logits, subcategory_logits

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_dataloader, val_dataloader, epochs, learning_rate, device):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    category_loss_fn = nn.CrossEntropyLoss()
    subcategory_loss_fn = nn.CrossEntropyLoss()
    
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(train_dataloader):
            input_ids, y_cat, y_sub = [item.to(device) for item in batch]
            
            optimizer.zero_grad()
            
            cat_logits, sub_logits = model(input_ids, attention_mask=(input_ids != 0))
            
            cat_loss = category_loss_fn(cat_logits, y_cat)
            sub_loss = subcategory_loss_fn(sub_logits, y_sub)
            
            loss = cat_loss + sub_loss
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Train Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, y_cat, y_sub = [item.to(device) for item in batch]
                cat_logits, sub_logits = model(input_ids, attention_mask=(input_ids != 0))
                
                cat_loss = category_loss_fn(cat_logits, y_cat)
                sub_loss = subcategory_loss_fn(sub_logits, y_sub)
                
                loss = cat_loss + sub_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
