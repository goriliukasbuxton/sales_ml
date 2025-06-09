

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

import os

# ----------------------
# Load and preprocess
# ----------------------
dtype = {'state': 'string', 'salesyear': 'int64', 'lifecycle': 'string', 'units': 'int64'}
df1 = pd.read_csv('case_study_data.csv', dtype=dtype)
df2 = pd.read_csv('GRIDMET_PDSI_State_Year_Wide_2021_2025.csv')
df2 = df2[['state', 'year', '12_PDSI']]
corn = pd.read_csv('yearly_mean_corn_prices_2015_2025.csv')
soy = pd.read_csv('yearly_mean_soybean_prices_2015_2025.csv')

use_optional_data = True

df1.columns = df1.columns.str.lower()
df1['year'] = df1['salesyear']
df1['age_of_product'] = df1['salesyear'] - df1['release_year']
df1['drought_tolerance'] = df1['drought_tolerance'].fillna(df1['drought_tolerance'].mean())
df1['plant_height'] = df1['plant_height'].fillna(df1['plant_height'].mean())

if use_optional_data:
    merged_df = pd.merge(df1, df2, on=['state', 'year'], how='outer')
    merged_df = pd.merge(merged_df, corn, on='year', how='outer')
    merged_df = pd.merge(merged_df, soy, on='year', how='outer')
else:
    merged_df = df1.copy()

# Encode categorical
merged_df['state'] = merged_df['state'].astype('category')
merged_df['lifecycle'] = merged_df['lifecycle'].astype('category')
merged_df['year'] = merged_df['year'].astype(int)
state_mapping = {state: idx for idx, state in enumerate(merged_df['state'].cat.categories, start=1)}
merged_df['state'] = merged_df['state'].map(state_mapping)

# ----------------------
# Create classification target
# ----------------------
merged_df['units_binary'] = (merged_df['units'] > 0).astype(int)

# ----------------------
# Train-Test Split for Both Models
# ----------------------
features = merged_df.drop(columns=['units', 'product', 'units_binary'])
X = features
y_class = merged_df['units_binary']
y_reg_all = merged_df['units']

# Split for classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Split for regression only on non-zero units
non_zero_df = merged_df[merged_df['units'] > 0].copy()
X_reg = non_zero_df.drop(columns=['units', 'product', 'units_binary'])
y_reg = non_zero_df['units']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# ----------------------
# Stage 1: Classification
# ----------------------
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True, random_state=42)
clf.fit(X_train_cls, y_train_cls)

# ----------------------
# Stage 2: Regression (only on non-zero)
# ----------------------
reg = XGBRegressor(objective='reg:tweedie', tweedie_variance_power=1.5, enable_categorical=True, random_state=42)
reg.fit(X_train_reg, y_train_reg)

# ----------------------
# Prediction and Evaluation
# ----------------------

# Predict probability of sale
proba_sale = clf.predict_proba(X_test_cls)[:, 1]

# Predict expected units (on whole X_test_cls)
reg_units = reg.predict(X_test_cls)

# Multiply both to get final expected units
final_pred = proba_sale * reg_units

# Ground truth
actual_units = y_reg_all.loc[X_test_cls.index]

# ----------------------
# Evaluation
# ----------------------
mse = mean_squared_error(actual_units, final_pred)
rmse = np.sqrt(mse)
r2 = r2_score(actual_units, final_pred)
print(f"Two-Stage Model RMSE: {rmse:.2f}")
print(f"Two-Stage Model MSE: {mse:.2f}")
print(f"Two-Stage Model R²: {r2:.3f}")

# ----------------------
# Optional: Classification Performance
# ----------------------
y_pred_cls = clf.predict(X_test_cls)
acc = accuracy_score(y_test_cls, y_pred_cls)
auc = roc_auc_score(y_test_cls, proba_sale)
print(f"Classification Accuracy: {acc:.3f}")
print(f"Classification ROC-AUC: {auc:.3f}")

# ----------------------
# Plot Actual vs Predicted
# ----------------------
plt.figure(figsize=(10, 6))
plt.scatter(actual_units, final_pred, alpha=0.5)
plt.plot([actual_units.min(), actual_units.max()], [actual_units.min(), actual_units.max()], 'r--')
plt.xlabel('Actual Units Sold')
plt.ylabel('Predicted Units Sold')
plt.title('Actual vs Predicted Units (Two-Stage Model)')
plt.grid()
plt.tight_layout()
plt.show()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[indices], importances[indices])
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


plot_feature_importance(reg, X.columns, "Stage 2: Feature Importance")

# %%
