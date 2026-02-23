
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def handle_unknowns(df):
    print("🔍 Handling 'unknown' values...")
    
    # 1. Job & Education: Mode Imputation
    # We use the most common value because these are highly skewed
    for col in ['job', 'education','contact']:
        mode_val = df[df[col] != 'unknown'][col].mode()[0]
        df[col] = df[col].replace('unknown', mode_val)
        print(f"   - Imputed {col} with mode: {mode_val}")

    # 2. Contact: Replace with 'missing' or 'cellular'
    # 'unknown' in contact often means 'landline' or 'missing data'
    #df['contact'] = df['contact'].replace('unknown', 'other')

    # 3. Poutcome (Previous Outcome): 
    # 'unknown' here is actually meaningful (it means the person was never contacted)
    # We keep it but rename it to 'non_existent' to be clearer for the model
    df['poutcome'] = df['poutcome'].replace('unknown', 'never_contacted')

    return df

def preprocess_raw_data(input_path, output_dir, model_dir):
    print(f"📥 Reading original data from: {input_path}")
    df = pd.read_csv(input_path)

    # NEW STEP: Handle Unknowns
    df = handle_unknowns(df)

    # 1. Feature Engineering (The logic we developed)
    print("⚙️ Engineering new features...")
    df['call_efficiency'] = df['duration'] / (df['campaign'] + 1)
    df['total_contacts'] = df['campaign'] + (df['previous'] if 'previous' in df.columns else 0)

    # 2. Define Features and Target
    # Ensure this matches the columns your preprocessor expects
    X = df.drop(columns=['y', 'duration']) # Dropping duration to prevent data leakage
    y = df['y'].map({'yes': 1, 'no': 0})

    # 3. Load Preprocessor (Scalers/Encoders)
    # We load it to ensure consistency between training and future data
    preprocessor_path = os.path.join(model_dir, 'preprocessor.joblib')
    if os.path.exists(preprocessor_path):
        print("🔗 Applying existing preprocessor...")
        preprocessor = joblib.load(preprocessor_path)
    else:
        print("⚠️ Preprocessor not found! You may need to fit a new one.")
        return

    # 4. Transform Data
    X_transformed = preprocessor.transform(X)
    
    # 5. Handle Imbalance (SMOTE)
    print("⚖️ Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

    # 6. Save Processed Files
    print(f"💾 Saving processed data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Converting back to DF for CSV saving (Optional but helpful)
    all_features = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    X_resampled_df = pd.DataFrame(X_resampled, columns=all_features)
    
    X_resampled_df.to_csv(os.path.join(output_dir, 'X_train_balanced.csv'), index=False)
    pd.Series(y_resampled).to_csv(os.path.join(output_dir, 'y_train_balanced.csv'), index=False)

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    preprocess_raw_data(
        input_path='../data/raw/bank-full.csv',
        output_dir='../data/processed/',
        model_dir='../models/joblib/'
    )