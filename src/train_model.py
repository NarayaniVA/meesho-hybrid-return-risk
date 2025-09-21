import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load training data
train_df = pd.read_csv('data/buyer_features_train.csv')

exclude_cols = ['customer_id', 'risk_category', 'fraud_label']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X = train_df[feature_cols]
y = train_df['fraud_label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

print("Model trained on training data.")
