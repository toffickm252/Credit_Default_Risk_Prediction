# Import the needed libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load data 
print("🚀 Starting Data Preprocessing Pipeline...")
print("=" * 60)

# base_dir = os.getcwd()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'Data', 'creditcard.csv')
output_path = os.path.join(base_dir, 'Data', 'Cleaned_creditcard_new.csv')
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Handle potential encoding errors
try:
    transaction_df = pd.read_csv(data_path)  # read the data from the path
except FileNotFoundError:
    print(f"Error: File not found at {data_path}. Check your working directory.")
    exit()

print(f"📊 Class distribution before preprocessing:")
print(f"   Legitimate: {(transaction_df['Class'] == 0).sum():,}")
print(f"   Fraud:      {(transaction_df['Class'] == 1).sum():,}")

print(f"Data loaded successfully with {len(transaction_df)} records.")

# Outlier handling 
Q1 = transaction_df['Amount'].quantile(0.25)
Q3 = transaction_df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_before = ((transaction_df['Amount'] < lower_bound) | (transaction_df['Amount'] > upper_bound)).sum()
transaction_df['Amount'] = transaction_df['Amount'].clip(lower=lower_bound, upper=upper_bound)

print(f"   Amount IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"   Outliers capped: {outliers_before:,}")
print("✅ Outlier handling complete")

# feature Engineering 
# Feature 1: Amount_scaled (RobustScaler)
robust_scaler = RobustScaler()
transaction_df['Amount_scaled'] = robust_scaler.fit_transform(transaction_df[['Amount']])
print("   ✅ Feature 1: Amount_scaled (RobustScaler)")
# Why scale Amount? The 'Amount' feature is highly skewed with extreme outliers. Using RobustScaler helps to reduce the influence of outliers while scaling the data, which can improve model performance and convergence.


# Feature 2: Time_hour (hour of day from seconds)
transaction_df['Time_hour'] = (transaction_df['Time'] % 86400) / 3600
print("   ✅ Feature 2: Time_hour (hour of day, 0-24)")

# Feature 3: Amount_log (log-transformed amount to reduce skew)
transaction_df['Amount_log'] = np.log1p(transaction_df['Amount'])
print("   ✅ Feature 3: Amount_log (log1p transformation)")

# Feature 4: V14_V12_interaction (top 2 negative fraud correlators)
transaction_df['V14_V12_interaction'] = transaction_df['V14'] * transaction_df['V12']
print("   ✅ Feature 4: V14_V12_interaction (interaction term)")

# Feature 5: High_risk_score (composite of top negative fraud features)
transaction_df['High_risk_score'] = transaction_df['V17'] + transaction_df['V14'] + transaction_df['V12'] + transaction_df['V10']
print("   ✅ Feature 5: High_risk_score (V17 + V14 + V12 + V10)")

# Feature 6: V_magnitude (euclidean magnitude of all PCA features)
pca_cols = [f'V{i}' for i in range(1, 29)]
transaction_df['V_magnitude'] = np.sqrt((transaction_df[pca_cols] ** 2).sum(axis=1))
print("   ✅ Feature 6: V_magnitude (PCA euclidean magnitude)")

# Feature 7: Time_is_night (binary: transactions between midnight and 6 AM)
transaction_df['Time_is_night'] = (transaction_df['Time_hour'] >= 0) & (transaction_df['Time_hour'] < 6)
print("   ✅ Feature 7: Time_is_night (binary: transactions between midnight and 6 AM)")


# Drop original Amount and Time (replaced by engineered versions)
transaction_df.drop(columns=['Amount', 'Time'], inplace=True)
print(f"Dropped original Amount and Time columns")
print(f"Final column count: {transaction_df.shape[1]}")


# Train Test split before handling class imbalance
print("\n" + "=" * 60)
print(" Step 4: Train/Test Split...")

X = transaction_df.drop(columns=['Class'])
y = transaction_df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set:     {X_test.shape[0]:,} samples")
print(f"   Train fraud rate: {y_train.mean():.4%}")
print(f"   Test fraud rate:  {y_test.mean():.4%}")

# sacle features for modeling (StandardScaler for PCA features, RobustScaler for engineered features)
print("\n" + "=" * 60)
print("Step 5: Scaling Features...")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

scaler_path = os.path.join(models_dir, "feature_scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"StandardScaler fit on training data")
print(f"Scaler saved to: {scaler_path}")

# handle class imbalance with SMOTE on training data only
print("\n" + "=" * 60)
print("Handling Class Imbalance with SMOTE...")

print(f"Before SMOTE:")
print(f"Legitimate: {(y_train == 0).sum():,}")
print(f"Fraud:      {(y_train == 1).sum():,}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)  # type: ignore

print(f"After SMOTE:")
print(f"Legitimate:{(y_train_resampled == 0).sum():,}")
print(f"Fraud:{(y_train_resampled == 1).sum():,}")
print(f"Training set balanced with SMOTE")

# Final check before saving
print("\n" + "=" * 60)
print("Final dataset ready for modeling:")
print(f"Training set shape: {X_train_resampled.shape}")
print(f"Test set shape:     {X_test_scaled.shape}")
print(f"Training fraud rate: {y_train_resampled.mean():.4%}")

# save the preprocessed datasets
train_data = pd.concat([X_train_resampled, y_train_resampled], axis=1)
test_data = pd.concat([X_test_scaled, y_test], axis=1)  
# train_data.to_csv(os.path.join(models_dir, "train_data.csv"), index=False)
# test_data.to_csv(os.path.join(models_dir, "test_data.csv"), index=False)
train_data.to_csv(os.path.join(base_dir, "Data", "train_processed.csv"), index=False)
test_data.to_csv(os.path.join(base_dir, "Data", "test_processed.csv"), index=False)
print(f"Preprocessed training and test datasets saved to {os.path.join(base_dir, 'Data')}")
