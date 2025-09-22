import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs('results', exist_ok=True)

# Load datasets
consolidated_df = pd.read_csv('data/consolidated_train_dataset.csv')
buyer_df = pd.read_csv('data/buyer_features.csv')
seller_df = pd.read_csv('data/seller_features.csv')
region_df = pd.read_csv('data/region_features.csv')

# --- Pie charts for Risk Category Distribution ---

# 1. Buyers risk categories
plt.figure(figsize=(6,6))
buyer_df['risk_category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange', 'salmon'])
plt.title('Buyer Risk Category Distribution')
plt.ylabel('')
plt.savefig('results/buyer_risk_category_distribution.png')
plt.show()

# 2. Sellers risk proxy - based on seller return rate bins
seller_df['risk_level'] = pd.qcut(seller_df['seller_return_rate'], q=3, labels=['Low', 'Medium', 'High'])
plt.figure(figsize=(6,6))
seller_df['risk_level'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'gold', 'tomato'])
plt.title('Seller Return Rate Risk Distribution')
plt.ylabel('')
plt.savefig('results/seller_risk_distribution.png')
plt.show()

# 3. Regions risk proxy - based on region RTO rate bins

# Sort regions by region_rto_rate descending
region_df = region_df.sort_values('region_rto_rate', ascending=False).reset_index(drop=True)

# Assign risk levels: top 1 -> High, next 1 -> Medium, remaining 2 -> Low
risk_levels = ['High', 'Medium'] + ['Low'] * (len(region_df) - 2)
region_df['risk_level'] = risk_levels

# Now plot using this risk_level column (pie chart)
plt.figure(figsize=(6,6))
region_df['risk_level'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['tomato', 'gold', 'lightblue'])
plt.title('Region RTO Rate Risk Distribution')
plt.ylabel('')
plt.savefig('results/region_risk_distribution.png')
plt.show()


# --- Average Frequencies of Return Reasons (Buyer) ---
reason_cols = [c for c in buyer_df.columns if c.startswith('reason_')]
mean_reasons = buyer_df[reason_cols].mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=mean_reasons.values, y=mean_reasons.index, palette='Blues_r')
plt.xlabel('Average Frequency')
plt.title('Average Return Reason Frequencies (Buyers)')
plt.savefig('results/average_return_reason_frequencies.png')
plt.show()

# --- RandomForest Feature Importances ---

exclude_cols = ['buyer_id', 'seller', 'region', 'risk_category', 'fraud_label', 'composite_risk_score']
feature_cols = [c for c in consolidated_df.columns if c not in exclude_cols]

X = consolidated_df[feature_cols]
y = consolidated_df['fraud_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
indices = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=[feature_cols[i] for i in indices], palette='viridis')
plt.title('RandomForest Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.show()


# 3. Correlation heatmap

# Select numeric features for correlation
numeric_cols = [
    'buyer_return_ratio', 'seller_return_rate', 'seller_complaint_rate', 
    'region_rto_rate', 'courier_flag_rate', 'seller_deviation', 'composite_risk_score'
]

plt.figure(figsize=(10,8))
sns.heatmap(consolidated_df[numeric_cols].corr(), annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numeric Features')
plt.savefig('results/feature_correlation_heatmap.png')
plt.show()

