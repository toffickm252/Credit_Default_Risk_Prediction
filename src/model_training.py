import numpy as np
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

# Load the preprocessed data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_path = os.path.join(base_dir, "Data", "train_processed.csv")
test_data_path = os.path.join(base_dir, "Data", "test_processed.csv")
reports_dir = os.path.join(base_dir, "reports")
os.makedirs(reports_dir, exist_ok=True)

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']

print(f"Data loaded successfully!")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# split the test set into validation and final test sets (80/20 split)
X_val, X_test_final, y_val, y_test_final = train_test_split(
    X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
)

print(f"Validation set shape: {X_val.shape}")
print(f"Final test set shape: {X_test_final.shape}")    

# Check class distribution in training, validation, and test sets
print(f"\nClass distribution:")
print(f"Training set - Legitimate: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,} (Fraud rate: {y_train.mean():.4%})")
print(f"Validation set - Legitimate: {(y_val == 0).sum():,}, Fraud: {(y_val == 1).sum():,} (Fraud rate: {y_val.mean():.4%})")
print(f"Final test set - Legitimate: {(y_test_final == 0).sum():,}, Fraud: {(y_test_final == 1).sum():,} (Fraud rate: {y_test_final.mean():.4%})")

# --- SAMPLE training data for faster initial model comparison ---
X_train_quick, _, y_train_quick, _ = train_test_split(
    X_train, y_train,
    train_size=0.25,  # Use 25% for quick model selection
    random_state=42,
    stratify=y_train
)
print(f"\nUsing {len(X_train_quick):,} samples (25%) for initial model comparison")

# Defining the models to train (lower n_estimators for quick comparison)
models = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'X_train': X_train_quick,
        'X_val': X_val
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
        'X_train': X_train_quick,
        'X_val': X_val
    },
    'XGBoost': {
        'model': XGBClassifier(n_estimators=50, random_state=42, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        'X_train': X_train_quick,
        'X_val': X_val
    }
}

print(f"Models to train: {', '.join(models.keys())}")

# Training of the models
results = {}

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    print('='*50)
    
    model = config['model']
    X_train_data = config['X_train']
    X_val_data = config['X_val']
    
    # Train
    model.fit(X_train_data, y_train_quick)
    
    # Predict
    y_pred = model.predict(X_val_data)
    y_pred_proba = model.predict_proba(X_val_data)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))

    # Store results
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_pred_proba,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC-AUC': roc_auc,
    }

# models comparison
print("\n" + "="*70)
print("MODEL COMPARISON AND ANALYSIS")
print("="*70)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['Accuracy'] for model in results.keys()],
    'Precision': [results[model]['Precision'] for model in results.keys()],
    'Recall': [results[model]['Recall'] for model in results.keys()],
    'F1-Score': [results[model]['F1'] for model in results.keys()],
    'ROC-AUC': [results[model]['ROC-AUC'] for model in results.keys()]
})

# Sort by F1-Score (important for imbalanced datasets)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n--- Performance Metrics Comparison ---")
print(comparison_df.to_string(index=False))

# Identify best model for each metric
print("\n--- Best Model by Metric ---")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
    best_score = comparison_df[metric].max()
    print(f"{metric:<12}: {best_model:<25} ({best_score:.4f})")

# Overall best model (based on F1-Score)
best_overall = comparison_df.iloc[0]['Model']
print(f"\nOverall Best Model (F1-Score): {best_overall}")

# Save models summary to reports directory
summary_path = os.path.join(reports_dir, "models_summary.txt")
with open(summary_path, 'w') as f:
    f.write("="*90 + "\n")
    f.write("MODELS EVALUATION SUMMARY\n")
    f.write("="*90 + "\n\n")
    
    # Table header
    f.write(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}\n")
    f.write("-"*90 + "\n")
    
    # Table rows
    for name, result in results.items():
        roc_auc_str = f"{result['ROC-AUC']:.4f}" if result['ROC-AUC'] else "N/A"
        f.write(f"{name:<25} {result['Accuracy']:<12.4f} {result['Precision']:<12.4f} "
                f"{result['Recall']:<12.4f} {result['F1']:<12.4f} {roc_auc_str:<12}\n")
    
    f.write("="*90 + "\n")

print(f"✓ Models summary saved to {summary_path}")

# Hyperparameter tuning and final evaluation on test set
print("\n" + "="*70)
print("HYPERPARAMETER TUNING AND FINAL EVALUATION")
print("="*70)

# Select best model based on F1-Score
best_model_name = max(results.items(), key=lambda x: x[1]['F1'])[0]
print(f"\nBest model: {best_model_name}")
print(f"Validation F1-Score: {results[best_model_name]['F1']:.4f}")

# Define trimmed parameter distributions for each model
param_distributions = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
    },
    'Random Forest': {
        'n_estimators': [200, 300],
        'max_depth': [15, 20, 30],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt'],
        'bootstrap': [True],
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [3, 5]
    }
}

# Recreate a fresh base model with full n_estimators for tuning
base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
}

base_model = base_models[best_model_name]
param_dist = param_distributions[best_model_name]

# SAMPLE TRAINING DATA FOR FASTER HYPERPARAMETER SEARCH
print("\n" + "="*70)
print("SAMPLING TRAINING DATA FOR HYPERPARAMETER SEARCH")
print("="*70)

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train, y_train, 
    train_size=0.5,  # Use 50% of data
    random_state=42,
    stratify=y_train
)

print(f"Original training size: {len(X_train):,} samples")
print(f"Sampled training size: {len(X_train_sample):,} samples (50%)")
print(f"Fraud rate in sample: {y_train_sample.mean():.4%}")

random_search = HalvingRandomSearchCV(
    base_model,
    param_distributions=param_dist,
    n_candidates=20,   # start with 20 candidates
    factor=3,           # eliminate 2/3 each round
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print(f"\nRunning HalvingRandomSearchCV for {best_model_name}...")
print(f"Starting with 20 candidates, eliminating 2/3 each round")
random_search.fit(X_train_sample, y_train_sample)

print(f"\nBest parameters:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"Best CV F1 score: {random_search.best_score_:.4f}")

# RETRAIN ON FULL DATA
print("\n🔄 Retraining best model on FULL training data...")
tuned_model = random_search.best_estimator_  # type: ignore
tuned_model.fit(X_train, y_train)  # type: ignore
print("✅ Model retrained on full dataset")

# Evaluate on final test set
y_pred_tuned = tuned_model.predict(X_test_final)  # type: ignore
y_prob_tuned = tuned_model.predict_proba(X_test_final)[:, 1]  # type: ignore

tuned_f1 = f1_score(y_test_final, y_pred_tuned)
tuned_roc = roc_auc_score(y_test_final, y_prob_tuned)
tuned_prec = precision_score(y_test_final, y_pred_tuned)
tuned_rec = recall_score(y_test_final, y_pred_tuned)

print(f"\nTuned {best_model_name} on FINAL test set:")
print(f"  F1:        {tuned_f1:.4f}")
print(f"  ROC-AUC:   {tuned_roc:.4f}")
print(f"  Precision: {tuned_prec:.4f}")
print(f"  Recall:    {tuned_rec:.4f}")

print(f"\nClassification Report (Tuned {best_model_name}):")
print(classification_report(y_test_final, y_pred_tuned, target_names=['Legitimate', 'Fraud']))

# Create models directory and save tuned model
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, f"best_model_{best_model_name.replace(' ', '_').lower()}_tuned.joblib")
joblib.dump(tuned_model, model_path)
print(f"💾 Tuned model saved to: {model_path}")

# Save tuning report
tuning_path = os.path.join(reports_dir, "hyperparameter_tuning.txt")
with open(tuning_path, 'w') as f:
    f.write(f"HYPERPARAMETER TUNING - {best_model_name}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Best Parameters:\n")
    for param, value in random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nBest CV F1: {random_search.best_score_:.4f}\n")
    f.write(f"\nFinal Test Performance:\n")
    f.write(f"  F1: {tuned_f1:.4f}, ROC-AUC: {tuned_roc:.4f}\n")
    f.write(f"  Precision: {tuned_prec:.4f}, Recall: {tuned_rec:.4f}\n\n")
    f.write(str(classification_report(y_test_final, y_pred_tuned, target_names=['Legitimate', 'Fraud'])))
print(f"Tuning report saved to: {tuning_path}")