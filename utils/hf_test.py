from utils.hf_api_predict import BankTransactionCategorizerHF

categorizer = BankTransactionCategorizerHF(
    cat_repo='jonngan/trans-cat',
    subcat_repo='jonngan/trans-subcat'
)

# Example DataFrame
import pandas as pd
df = pd.DataFrame([{"Description": "Starbucks Coffee"}, {"Description": "Shell Gas Station"}])

result_df = categorizer.predict(df)
print(result_df)