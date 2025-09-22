import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load consolidated training dataset
train_df = pd.read_csv('data/consolidated_train_dataset.csv')

# Define features and label columns
exclude_cols = ['buyer_id', 'seller', 'region', 'risk_category', 'fraud_label', 'composite_risk_score']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X = train_df[feature_cols]
y = train_df['fraud_label']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

print("Model trained successfully.")
