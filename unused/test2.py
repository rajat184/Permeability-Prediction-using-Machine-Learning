from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# from sklearn.datasets import fetch_california_housing

# boston = fetch_california_housing()

data_url = "https://pubs.acs.org/doi/suppl/10.1021/acscentsci.8b00718/suppl_file/oc8b00718_si_002.txt"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=28, header=None)   # this is x that is input
# data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])    # this is y that is output 
# boston = raw_df.values[1::2, 2]


# Read the dataset string into a pandas DataFrame
df = pd.read_csv(data_url, sep=" ", header=None)

# Assign column names
df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

# Extracting features (X) and target variable (Y)
X = df[["Col3", "Col4", "Col5", "Col6", "Col7"]]
y = df["Col8"]


# X, y = shuffle(boston.data, boston.target, random_state=13)
# X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

