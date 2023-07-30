# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt

# Step 2: Preprocess the data
# Load the training data
train_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Signate/test.csv')

# Load the test data
test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Signate/train.csv')

# Combine train and test data to preprocess them together
combined_data = pd.concat([train_data, test_data])

# Correct anomalies
combined_data.loc[combined_data['year'] >= 3000, 'year'] -= 1000  # adjust years in 3000s to 2000s
combined_data['odometer'] = combined_data['odometer'].abs()  # take absolute value of odometer

# Fill numerical missing values with KNN imputation
imputer = KNNImputer(n_neighbors=5)
combined_data[['odometer', 'year']] = imputer.fit_transform(combined_data[['odometer', 'year']])

# Calculate vehicle age
current_year = datetime.now().year
combined_data['vehicle_age'] = current_year - combined_data['year']

# Calculate mileage per year
combined_data['mileage_per_year'] = combined_data['odometer'] / combined_data['vehicle_age']

# Fill categorical missing values with 'unknown'
categorical_features = ['region', 'manufacturer', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'state']
combined_data[categorical_features] = combined_data[categorical_features].fillna('unknown')

# One-hot encode categorical features
combined_data = pd.get_dummies(combined_data, columns=categorical_features)

# Scale numerical features
numerical_features = ['odometer', 'year', 'vehicle_age', 'mileage_per_year']
scaler = StandardScaler()
combined_data[numerical_features] = scaler.fit_transform(combined_data[numerical_features])

# Separate preprocessed train and test data
train_data = combined_data[combined_data['price'].notna()]
test_data = combined_data[combined_data['price'].isna()]

# Split the data based on vehicle type
vehicle_types = ["type_truck", "type_bus", "type_pickup"]
train_data_type1 = train_data[train_data[vehicle_types].any(axis=1)]
train_data_type2 = train_data[~train_data[vehicle_types].any(axis=1)]

test_data_type1 = test_data[test_data[vehicle_types].any(axis=1)]
test_data_type2 = test_data[~test_data[vehicle_types].any(axis=1)]

# Step 3: Train Neural Network models and make predictions
# Define KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=1)

# Define learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * 1

callback = LearningRateScheduler(scheduler)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Create and fit models with KFold cross-validation
test_predictions = []
train_datasets = [(train_data_type1, test_data_type1), (train_data_type2, test_data_type2)]

for train_data, test_data in train_datasets:
    features = train_data.drop(columns=['price', 'id'])
    target = train_data['price']
    test_features = test_data.drop(columns=['price', 'id'])

    for train_index, val_index in kfold.split(features):
        train_features, val_features = features.iloc[train_index], features.iloc[val_index]
        train_target, val_target = target.iloc[train_index], target.iloc[val_index]

        # Create a neural network model with dropout
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=train_features.shape[1]))
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

        model.compile(optimizer=Adam(0.0001), loss='mean_absolute_percentage_error')

        # Fit the model with more epochs
        history = model.fit(train_features, train_target, epochs=10000, batch_size=256, verbose=1,
                            callbacks=[callback, early_stopping], validation_data=(val_features, val_target))

        # After fitting, plot the loss per epoch
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions and store them
        model_predictions = model.predict(test_features)
        test_predictions.append((test_data['id'], model_predictions))

# Step 4: Save predictions to a CSV file
# Concatenate predictions from all models
test_predictions = pd.concat([pd.DataFrame({'id': ids, 'price': preds.flatten()}) for ids, preds in test_predictions])

# Average predictions for each ID
test_predictions = test_predictions.groupby('id').mean().reset_index()

date_string = datetime.now().strftime("%Y-%m-%d")
test_predictions.to_csv('/content/drive/MyDrive/Colab Notebooks/Signate/submission_{}.csv'.format(date_string), index=False, header=False)
