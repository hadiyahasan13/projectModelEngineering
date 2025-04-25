#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score


# In[3]:


get_ipython().system('pip install imblearn')


# In[2]:


# Load dataset
df = pd.read_csv("C:\\Users\\Hp\\Desktop\\creditcard\\creditcard.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


print("Number of columns: {}".format(df.shape[1]))
print("Number of rows: {}".format(df.shape[0]))


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df.info


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
float_cols = [col for col in df.columns if df[col].dtype == "float64" or df[col].dtype == "int64"]

cols_per_row = 3
num_plots = len(float_cols)
rows = (num_plots // cols_per_row) + (num_plots % cols_per_row > 0) 

fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows)) 
axes = axes.flatten()  

for idx, col in enumerate(float_cols):
    sns.histplot(df[col], bins=50, kde=True, ax=axes[idx])
    axes[idx].set_title(f"Distribution of {col}")

for i in range(idx + 1, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# In[11]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Class')
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()


# In[12]:


fraud = df[df['Class'] == 1]
fraud_num = len(fraud)
valid = df[df['Class'] == 0]
valid_num = len(valid)
outlier_percentage = len(fraud) / float(len(valid)) * 100.0

print(f"{outlier_percentage:.2f}%")
print(f'Fraud transactions: {fraud_num}')
print(f'Valid transactions: {valid_num}')


# In[13]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Class', y='Amount')
plt.title('Transaction Amount, fraud vs non-fraud')
plt.xlabel('Class')
plt.ylabel('Transaction Amount')
plt.show()


# In[14]:


plt.scatter(fraud['Time'], fraud['Amount'], alpha=0.8)
plt.title('Fraudulent transactions over time')
plt.xlabel('Time')
plt.ylabel('Transaction Amount')
plt.show()


# In[15]:


fraud['Amount'].describe()


# In[16]:


corr_mat = df.corr()
fig = plt.figure(figsize=(15, 15))
sns.heatmap(data=corr_mat, annot=True, vmin=-1, vmax=1, fmt='.1f', square=True)
plt.title("Correlation Matrix", size=15)
plt.show()


# In[17]:


sc = StandardScaler()
df['Amount'] = sc.fit_transform(pd.DataFrame(df['Amount']))


# In[18]:


df.head()


# In[19]:


df = df.drop(['Time'], axis =1)


# In[20]:


df.head()


# In[21]:


df.duplicated().any()


# In[22]:


df = df.drop_duplicates()


# In[23]:


df.shape


# In[24]:


df['Class'].value_counts()


# In[25]:


# Split features and target
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target (0 = Legit, 1 = Fraud)



# In[26]:


from xgboost import XGBClassifier
# Define XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)


# In[27]:


from sklearn.model_selection import cross_val_predict, StratifiedKFold
# 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[28]:


from sklearn.model_selection import cross_val_predict, StratifiedKFold
# Cross-validation predictions
y_pred = cross_val_predict(xgb, X, y, cv=cv)


# In[29]:


from sklearn.metrics import classification_report
# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))


# In[30]:


# Hyperparameter grid for tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


# In[31]:


# 5-fold cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[32]:


# Randomized Search with 10 iterations
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,  # Limits number of parameter combinations
    scoring="f1",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)


# In[33]:


# Fit RandomizedSearchCV
random_search.fit(X, y)


# In[34]:


# Get best parameters
best_xgb = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)


# In[35]:


# Evaluate best model using 5-Fold CV predictions
y_pred = cross_val_predict(best_xgb, X, y, cv=cv)


# In[36]:


# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))


# In[37]:


# Check class distribution before applying SMOTE
print("Class distribution before SMOTE:\n", y.value_counts())


# In[38]:


# Apply SMOTE only on the training folds (to prevent data leakage)
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust ratio if needed


# In[39]:


from imblearn.pipeline import Pipeline as ImbPipeline
# Define a pipeline with SMOTE and XGBoost
pipeline = ImbPipeline([("smote", smote), ("xgb", xgb)])


# In[40]:


# Randomized Search with 10 iterations
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={"xgb__" + key: value for key, value in param_grid.items()},
    n_iter=10,  # Limits number of parameter combinations
    scoring="f1",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)


# In[41]:


# Fit RandomizedSearchCV
random_search.fit(X, y)



# In[42]:


# Get best parameters
best_xgb = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)


# In[43]:


# Evaluate best model using 5-Fold CV predictions
y_pred = cross_val_predict(best_xgb, X, y, cv=cv)


# In[44]:


# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))


# In[45]:


# Iterate through folds and apply SMOTE inside each fold
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Print class distribution before SMOTE
    print(f"Fold {fold+1} - Class distribution before SMOTE:\n{y_train.value_counts()}")

    # Apply SMOTE only on training data (not validation)
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Print class distribution after SMOTE
    print(f"Fold {fold+1} - Class distribution after SMOTE:\n{y_train_resampled.value_counts()}")


# In[46]:


# Define pipeline with SMOTE and XGBoost
pipeline = ImbPipeline([("smote", SMOTE(sampling_strategy=0.5, random_state=42)), ("xgb", xgb)])


# In[47]:


# Randomized Search with 10 iterations
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions={"xgb__" + key: value for key, value in param_grid.items()},
    n_iter=10,  # Limits number of parameter combinations
    scoring="f1",
    n_jobs=-1,
    cv=cv,
    verbose=1,
    random_state=42
)


# In[48]:


# Fit RandomizedSearchCV
random_search.fit(X, y)


# In[49]:


# Get best parameters
best_xgb = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Evaluate best model using 5-Fold CV predictions
y_pred = cross_val_predict(best_xgb, X, y, cv=cv)

# Print classification report
print("\nClassification Report:")
print(classification_report(y, y_pred))


# In[ ]:




