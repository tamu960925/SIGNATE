import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import hashlib
import optuna

# Load data
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Signate/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Signate/test.csv')

# Preprocess function
def preprocess(df):
    df = df.fillna('-1')
    for c in df.columns[df.dtypes == 'object']:
        df[c] = df[c].astype(str)
        df[c] = LabelEncoder().fit_transform(df[c])
    return df

# Preprocess data
train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Split features and target
X = train_df.drop(['id', 'price'], axis=1)
y = train_df['price']
X_test = test_df.drop('id', axis=1)

# Initialize KFold
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Initialize an empty array for predictions
predictions = np.zeros(len(X_test))

# Define objective function for Optuna
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 700, 3000),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'num_leaves': trial.suggest_int('num_leaves', 100, 500),
        'min_child_samples': trial.suggest_int('min_child_samples', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    }
    
    model = LGBMRegressor(**params)
    
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100), log_evaluation(100)])
        
    y_pred = model.predict(X_val)
    return mean_absolute_percentage_error(y_val, y_pred)

# Initialize Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Train and validate model
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = LGBMRegressor(**study.best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100), log_evaluation(100)])
    predictions += model.predict(X_test) / kf.n_splits

# Create submission dataframe
submission = test_df[['id']].copy()
submission['price'] = predictions

# Save submission file with a hash filename
submission_filename = hashlib.sha256(submission.to_csv(index=False).encode()).hexdigest() + '.csv'
submission.to_csv('/content/drive/MyDrive/Colab Notebooks/Signate/' + submission_filename, index=False)
