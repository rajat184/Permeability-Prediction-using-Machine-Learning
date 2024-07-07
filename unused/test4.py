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

# Split the data into training and testing sets
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Initialize LazyRegressor and fit the models
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)




