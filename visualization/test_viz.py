import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
os.makedirs('results', exist_ok=True)


# Load test data with predictions
test_df = pd.read_csv('test_risk_predictions.csv')

# 1. Prediction distribution histogram (KDE)
plt.figure(figsize=(8,4))
sns.kdeplot(test_df['predicted_risk'], shade=True, color='skyblue')
plt.axvline(0.5, color='red', linestyle='--', label='Flag Threshold')
plt.title('Predicted Risk Score Distribution (Test)')
plt.xlabel('Predicted Risk Score')
plt.legend()
plt.savefig('results/test_predicted_risk_distribution.png')
plt.show()

# 2. Confusion matrix heatmap
if 'true_label' in test_df.columns:
    cm = confusion_matrix(test_df['true_label'], test_df['flagged'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test)')
    plt.savefig('results/test_confusion_matrix.png')
    plt.show()

# 3. Flagged vs genuine counts bar chart
counts = test_df['flagged'].value_counts().rename({0:'Genuine/Not flagged',1:'Flagged'})
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values, palette='muted')
plt.title('Flagged vs Genuine Returns (Test)')
plt.ylabel('Count')
plt.savefig('results/test_flagged_vs_genuine_counts.png')
plt.show()
