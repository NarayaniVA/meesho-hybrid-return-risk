import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load train data
train_df = pd.read_csv('buyer_features_train.csv')

# 1. Class distribution pie chart
plt.figure(figsize=(6,6))
train_df['risk_category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Train Data Risk Category Distribution')
plt.ylabel('')
plt.show()


# 2. Return reason frequency bar plot (mean per reason)
reason_cols = [c for c in train_df.columns if c.startswith('reason_')]
mean_reasons = train_df[reason_cols].mean().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=mean_reasons.values, y=mean_reasons.index, palette='Blues_r')
plt.xlabel('Average Frequency')
plt.title('Average Return Reason Frequencies (Train)')
plt.show()

# 3. Correlation heatmap of numeric features
plt.figure(figsize=(12,10))
sns.heatmap(train_df.select_dtypes(include='number').corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap (Train)')
plt.show()

# 4. Feature importance bar plot from RandomForest model (requires trained model rf_model and feature_cols list)
importances = rf_model.feature_importances_
features = feature_cols

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title('RandomForest Feature Importance (Train)')
plt.show()
