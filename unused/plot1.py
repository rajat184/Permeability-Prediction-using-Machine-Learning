import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from lazypredict.Supervised import LazyRegressor
from sklearn.utils import shuffle
import pandas as pd

# Load dataset from CSV file
url1 = 'C:/Users/Suraj/Desktop/IP sem 6/test/Codes/Lazy Predict/input_data.csv'
df1 = pd.read_csv(url1)
df2 = pd.read_csv("C:/Users/Suraj/Desktop/IP sem 6/test/Codes/Lazy Predict/descriptors.csv")

X = df2.iloc[1:, :]  # Features
# the target variable is in the last column
y = df1.iloc[1:, -1]   # Target variable

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Assuming X and y are the features and target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the data into training and testing sets
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

from sklearn.impute import SimpleImputer

# Create an imputer object with a strategy of mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the train set
imputer = imputer.fit(X_train)

# Transform the training and testing set
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

# Train the ExtraTreesRegressor model
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values (ExtraTreesRegressor)')
plt.show()

# Print R^2 score
print(f'R^2 Score: {r2_score(y_test, y_pred):.2f}')




import pandas as pd

# Feature importance
feature_importances = model.feature_importances_
features = X.columns

# Create a DataFrame
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot

# Sort the features based on their importance
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

# Select top N features
top_n = 10
top_features = importance_df_sorted[:top_n]

# Plot
plt.figure(figsize=(10, 5))
plt.barh(top_features['Feature'], top_features['Importance'], color='b')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top {} Feature Importance (ExtraTreesRegressor)'.format(top_n))
plt.gca().invert_yaxis()

plt.show()