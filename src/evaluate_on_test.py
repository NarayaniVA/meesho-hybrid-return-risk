import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Load training data and test data
train_df = pd.read_csv('data/buyer_features_train.csv')
test_df = pd.read_csv('data/test_return_requests_noisy.csv')

exclude_cols = ['customer_id', 'risk_category', 'fraud_label']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

# Prepare train data for scaler and model recreation
X_train = train_df[feature_cols]
y_train = train_df['fraud_label']

# Prepare test data
X_test = test_df[feature_cols]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model on full training data (again since no saved model)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict test probabilities
test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
test_preds = (test_probs > 0.5).astype(int)

test_df['predicted_risk'] = test_probs
test_df['flagged'] = test_preds

print("Test Classification Report:")
print(classification_report(test_df['true_label'], test_preds))
print("Test ROC-AUC Score:", roc_auc_score(test_df['true_label'], test_probs))

test_df[['test_order_id', 'customer_id', 'return_reason', 'predicted_risk', 'flagged', 'true_label']].to_csv('test_risk_predictions.csv', index=False)
print("Saved test risk predictions to 'test_risk_predictions.csv'")
