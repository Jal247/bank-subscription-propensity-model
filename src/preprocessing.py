import pandas as pd

def clean_bank_data(df):
    """
    Professional cleaning based on Chi-Square results:
    - Impute job/education (low unknowns) with Mode.
    - Keep contact/poutcome (high unknowns) because they are significant.
    """
    df_clean = df.copy()

    # 1. Impute low-percentage unknowns
    for col in ['job', 'education', 'contact']:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].replace('unknown', mode_val)

    # 2. Convert Target to Numeric (Pro Move)
    if 'y' in df_clean.columns:
        df_clean['target'] = df_clean['y'].map({'yes': 1, 'no': 0})
    
    return df_clean
