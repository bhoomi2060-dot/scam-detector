# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:26:50 2026

@author: Admin
"""

# ==============================
# IMPORT LIBRARIES
# ==============================

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, confusion_matrix
)

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("C:/Users/shivani sharma/Downloads/Dataset/iris.data", header=None)

df.columns = ['sepal_length','sepal_width','petal_length','petal_width','class']

print("First 5 rows of dataset")
print(df.head())

# ==============================
# 1. MISSING VALUES
# ==============================

print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

# ==============================
# 2. REMOVE DUPLICATES
# ==============================

print("\nDuplicate rows before removing:", df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicate rows after removing:", df.duplicated().sum())

# ==============================
# 3. STATISTICAL MEASURES
# ==============================

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nMode:")
print(df.mode())

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

print("\nVariance:")
print(df.var(numeric_only=True))

# ==============================
# SEPARATE FEATURES & TARGET
# ==============================

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# ==============================
# 4. RESHAPING
# ==============================

X_reshaped = X.reshape(-1,4)
print("\nReshaped Data Shape:", X_reshaped.shape)

# ==============================
# 5. MIN-MAX SCALING
# ==============================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_reshaped)

print("\nNormalized Data (First 5 rows):")
print(X_scaled[:5])

# ==============================
# ENCODE TARGET
# ==============================

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==============================
# 6. PCA
# ==============================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nShape after PCA:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ==============================
# TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_encoded, test_size=0.3, random_state=42
)

# ==============================
# LOGISTIC REGRESSION
# ==============================

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, average='macro'))
print("Recall:", recall_score(y_test, y_pred_lr, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_lr, average='macro'))
print("AUC:", roc_auc_score(y_test, y_prob_lr, multi_class='ovr'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# ==============================
# DECISION TREE
# ==============================

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt, average='macro'))
print("Recall:", recall_score(y_test, y_pred_dt, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_dt, average='macro'))
print("AUC:", roc_auc_score(y_test, y_prob_dt, multi_class='ovr'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# ==============================
# RANDOM FOREST (ENSEMBLE)
# ==============================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, average='macro'))
print("Recall:", recall_score(y_test, y_pred_rf, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_rf, average='macro'))
print("AUC:", roc_auc_score(y_test, y_prob_rf, multi_class='ovr'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ==============================
# GRADIENT BOOSTING (ENSEMBLE)
# ==============================

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
y_prob_gb = gb_model.predict_proba(X_test)

print("\n=== Gradient Boosting Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
print("Precision:", precision_score(y_test, y_pred_gb, average='macro'))
print("Recall:", recall_score(y_test, y_pred_gb, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred_gb, average='macro'))
print("AUC:", roc_auc_score(y_test, y_prob_gb, multi_class='ovr'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_gb))

# ==============================
# POLYNOMIAL LINEAR REGRESSION
# ==============================

X_reg = df[['sepal_length', 'sepal_width']].values
y_reg = df['petal_length'].values

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_reg)
X_test_poly = poly.transform(X_test_reg)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_reg)

y_pred_reg = poly_model.predict(X_test_poly)

print("\n=== Polynomial Linear Regression Results ===")
print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))