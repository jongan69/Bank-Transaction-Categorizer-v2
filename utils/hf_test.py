from .hf_api_predict import BankTransactionCategorizerHF

categorizer = BankTransactionCategorizerHF()

# Example DataFrame
import pandas as pd
df = pd.DataFrame([{"Description": "Starbucks Coffee"}, {"Description": "Shell Gas Station"}])

result_df = categorizer.predict(df)
print(result_df)