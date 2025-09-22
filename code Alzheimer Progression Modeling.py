# ==============================================================
# Alzheimer’s Progression Modeling
# Submitted by: [Saurabh Kumar Kanth/ 2320667]
# ==============================================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Step 1: Generate Mock dtset
# -----------------------------
np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.randint(55, 90, 200),                  # age is taken in years
    "cognitive_score": np.random.randint(15, 30, 200),     # cognitive assessment scr
    "mri_metric": np.random.normal(2.5, 0.3, 200),         # MRI-derived brain metric taken from kaggle and modified 
    "progression_score": np.random.normal(0, 1, 200)       # tgt variable
})

# mock dataset saving in csv format
df.to_csv("alzheimer_progression_dataset.csv", index=False)
print("Mock dataset saved as 'alzheimer_progression_dataset.csv'\n")

# -----------------------------
# Step 2: Loading Dtset
# -----------------------------
df = pd.read_csv("alzheimer_progression_dataset.csv")
print("Dataset Head:")
print(df.head(), "\n")

# -----------------------------
# Step 3: Data analysis
# -----------------------------
print("Dataset Info:")
print(df.info(), "\n")
print("Dataset Description:")
print(df.describe(), "\n")

# heatmap cde
plt.figure(figsize=(6,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# dtribution of target variable
plt.figure(figsize=(6,4))
sns.histplot(df['progression_score'], kde=True, color='purple')
plt.title("Progression Score Distribution")
plt.show()

# -----------------------------
# Step 4: selection & pipeline feature
# -----------------------------
X = df.drop("progression_score", axis=1)
y = df["progression_score"]

# Split in trn set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pipline creation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression, k='all')),  # can adjust k
    ('regressor', LinearRegression())
])

# model train
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# -----------------------------
# Step 5: Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R^2 Score: {r2:.4f}\n")

# Feature importance from SelectKBest
feature_scores = pipeline.named_steps['feature_selection'].scores_
features_df = pd.DataFrame({
    "Feature": X.columns,
    "Score": feature_scores
}).sort_values(by='Score', ascending=False)
print("Feature Scores (Importance):")
print(features_df, "\n")

# -----------------------------
# Step 6: Visualization
# -----------------------------
# Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Progression Score")
plt.ylabel("Predicted Progression Score")
plt.title("Actual vs Predicted Progression")
plt.show()

# Feature importance bar plot
plt.figure(figsize=(6,4))
sns.barplot(x='Score', y='Feature', data=features_df, palette='viridis')
plt.title("Feature Importance")
plt.show()
