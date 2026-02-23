import pandas as pd
import numpy as np
import joblib
import pickle
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def run_full_pipeline():
    print("🚀 Starting End-to-End Pipeline...")

    # --- STEP 1: SETUP PATHS ---
    # Adjust paths if running from within /src/ or the root folder
    paths = {
        'raw_data': '../data/raw/bank-full.csv',
        'processed_dir': '../data/processed/',
        'model_dir': '../models/',
        'pickle_dir': '../data/pickle/'
    }
    
    # Ensure directories exist
    for folder in paths.values():
        if '.' not in folder: # Skip files, check directories
            os.makedirs(folder, exist_ok=True)

    # --- STEP 2: DATA LOADING & PREPROCESSING ---
    print("📥 Loading and Preprocessing Data...")
    # Load your processed datasets (assuming they were saved previously)
    X_train = pd.read_csv(os.path.join(paths['processed_dir'], 'X_train_resampled.csv'))
    y_train = pd.read_csv(os.path.join(paths['processed_dir'], 'y_train_resampled.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(paths['processed_dir'], 'X_test_final.csv'))
    y_test = pd.read_csv(os.path.join(paths['processed_dir'], 'y_test_final.csv')).values.ravel()
    
    # Load the preprocessor and feature names saved from your notebook
    preprocessor = joblib.load(os.path.join(paths['model_dir'], 'preprocessor.joblib'))
    all_feature_names = joblib.load(os.path.join(paths['model_dir'], 'feature_names.joblib'))

    # --- STEP 3: MODEL TRAINING ---
    print("🏋️  Training Optimized XGBoost Model...")
    # Using the best parameters discovered during your research
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # --- STEP 4: EVALUATION & VISUALIZATION ---
    print("📊 Evaluating Performance...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Print Report
    print(classification_report(y_test, y_pred))

    # Save Confusion Matrix Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(paths['processed_dir'], 'confusion_matrix.png'))
    plt.close()

    # --- STEP 5: INTERPRETABILITY (SHAP) ---
    print("🔍 Generating SHAP Interpretability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Save SHAP Summary Plot
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_test, feature_names=all_feature_names, show=False)
    plt.savefig(os.path.join(paths['processed_dir'], 'shap_summary.png'))
    plt.close()

    # --- STEP 6: SAVING ARTIFACTS ---
    print("💾 Saving Models, Pickles, and Requirements...")
    
    # Save Model
    joblib.dump(model, os.path.join(paths['model_dir'], 'xgboost_term_deposit_model.joblib'))
    
    # Save Data as Pickles
    datasets = {'X_test': X_test, 'y_test': y_test, 'shap_values': shap_values}
    with open(os.path.join(paths['pickle_dir'], 'deployment_data.pkl'), 'wb') as f:
        pickle.dump(datasets, f)
        
    # Update Requirements
    os.system('pip freeze > ../requirements.txt')

    print("\n✅ Pipeline complete. All files saved to /models, /data, and /processed.")

if __name__ == "__main__":
    run_full_pipeline()