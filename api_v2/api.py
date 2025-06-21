from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from utils.api_predict import BankTransactionCategorizerHF
from datetime import datetime

app = FastAPI()

class Transaction(BaseModel):
    Description: str

class TransactionsRequest(BaseModel):
    transactions: List[Transaction]

class CategorizedTransaction(BaseModel):
    Description: str
    Category: str
    Subcategory: str

class CategorizedResponse(BaseModel):
    results: List[CategorizedTransaction]

# Initialize the categorizer once (loads models from HF Hub)
categorizer = BankTransactionCategorizerHF()

@app.post("/categorize", response_model=CategorizedResponse)
def categorize_transactions(request: TransactionsRequest):
    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided.")

    log_file_path = os.path.join(os.path.dirname(__file__), "data", "transaction_requests.csv")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    descriptions = [t.Description for t in request.transactions]
    
    log_df = pd.DataFrame({
        "timestamp": [timestamp] * len(descriptions),
        "Description": descriptions
    })

    file_exists = os.path.isfile(log_file_path)
    log_df.to_csv(log_file_path, mode='a', header=not file_exists, index=False)

    # Prepare DataFrame
    df = pd.DataFrame([t.dict() for t in request.transactions])
    # Predict
    results_df = categorizer.predict(df)
    # Build response
    results = [CategorizedTransaction(
        Description=row["Description"],
        Category=row["Category"],
        Subcategory=row["Subcategory"]
    ) for _, row in results_df.iterrows()]
    return CategorizedResponse(results=results) 