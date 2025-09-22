import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Load consolidated training and noisy test datasets
train_df = pd.read_csv('data/consolidated_train_dataset.csv')
test_df = pd.read_csv('data/test_noisy_dataset.csv')

exclude_cols = ['buyer_id', 'seller', 'region', 'risk_category', 'fraud_label', 'composite_risk_score']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df['fraud_label']
X_test = test_df[feature_cols]
y_test = test_df['fraud_label']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test_scaled)
y_prob = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Save predictions
test_df['predicted_label'] = y_pred
test_df['predicted_proba'] = y_prob
test_df.to_csv('test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'")
