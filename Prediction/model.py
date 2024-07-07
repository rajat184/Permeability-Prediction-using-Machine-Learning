import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset from CSV file
url1 = 'C:/Users/Suraj/Desktop/IP sem 6/test/Codes/Lazy Predict/input_data.csv'
df1 = pd.read_csv(url1)
df2 = pd.read_csv("C:/Users/Suraj/Desktop/IP sem 6/test/Codes/Lazy Predict/descriptors.csv")

X = df2.iloc[1:, :]  # Features
y = df1.iloc[1:, -1]   # Target variable

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Split the data into training and testing sets
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Create an imputer object with a strategy of mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the train set
# imputer = fit(X_train)
# Fit the imputer to the train set
imputer = imputer.fit(X_train)

# Transform the training and testing set
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


from sklearn.metrics import r2_score

def train_and_plot(model, model_name):
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted Values ({model_name})')
    plt.show()

    # Print R^2 score
    print(f'R^2 Score for {model_name}: {r2_score(y_test, y_pred):.2f}')

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        features = X.columns

        # Create a DataFrame
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot
        top_n = 10
        top_features = importance_df[:top_n]

        plt.figure(figsize=(10, 5))
        plt.barh(top_features['Feature'], top_features['Importance'], color='b')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance ({model_name})')
        plt.gca().invert_yaxis()
        plt.show()
    else:
        print(f'{model_name} does not support feature importances.')

# Example usage with ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesRegressor
etr_model = ExtraTreesRegressor(random_state=42)
train_and_plot(etr_model, 'ExtraTreesRegressor')


from lightgbm import LGBMRegressor
lgb_model = LGBMRegressor(random_state=42)
train_and_plot(lgb_model, 'LGBMRegressor')


from sklearn.ensemble import HistGradientBoostingRegressor
hgb_model = HistGradientBoostingRegressor(random_state=42)
train_and_plot(hgb_model, 'HistGradientBoostingRegressor')


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)
train_and_plot(rf_model, 'RandomForestRegressor')


from xgboost import XGBRegressor
xgb_model = XGBRegressor(random_state=42)
train_and_plot(xgb_model, 'XGBRegressor')


