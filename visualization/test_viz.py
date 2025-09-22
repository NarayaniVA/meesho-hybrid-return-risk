import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

os.makedirs('results', exist_ok=True)

# Load test data with predictions
test_df = pd.read_csv('results/test_predictions.csv')

# 1. Predicted risk probability distribution
plt.figure(figsize=(8,4))
sns.kdeplot(test_df['predicted_proba'], shade=True, color='skyblue')
plt.axvline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Predicted Fraud Probability Distribution (Test)')
plt.xlabel('Predicted Fraud Probability')
plt.legend()
plt.savefig('results/test_predicted_probability_distribution.png')
plt.show()

# 2. Confusion matrix heatmap (requires true labels)
if 'fraud_label' in test_df.columns:
    cm = confusion_matrix(test_df['fraud_label'], test_df['predicted_label'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test)')
    plt.savefig('results/test_confusion_matrix.png')
    plt.show()

# 3. Flagged (predicted fraud) vs Genuine counts bar chart
counts = test_df['predicted_label'].value_counts().rename({0:'Genuine / Not flagged', 1:'Flagged / Fraud'})
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values, palette='muted')
plt.title('Flagged vs Genuine Return Counts (Test)')
plt.ylabel('Count')
plt.savefig('results/test_flagged_vs_genuine_counts.png')
plt.show()
