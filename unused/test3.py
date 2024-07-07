from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

# data_url = "oc8b00718_si_002.txt"

from io import StringIO

# Provided dataset as a string
dataset_str = """
CC C5 -1.977 -1.656 NOT NOT N 0.269
CN P2 1.455 0.920 NOT 10.08 B -4.686
CO P3 1.894 2.106 15.78 -2.46 A -2.865
CF Na -0.549 -0.595 NOT NOT N -1.005
FF N0 -1.304 -1.009 NOT NOT N -0.191
C=C N0 -1.235 -1.009 NOT NOT N -0.191
C=O P2 0.947 0.920 NOT -8.09 N -2.012
O=O P3 1.784 2.106 NOT NOT N -2.865
C#C P1 0.041 0.540 NOT NOT N -1.574
CCC C3 -3.006 -2.930 NOT NOT N 0.023
CCN P1 0.275 0.540 NOT 10.23 B -4.391
CCO P1 0.549 0.540 16.47 -2.16 N -1.574
CCF C5 -1.386 -1.656 NOT NOT N 0.269
FCF N0 -0.398 -1.009 NOT NOT N -0.191
CNC P1 0.727 0.540 NOT 10.52 B -4.678
COC P1 0.220 0.540 NOT -4.08 N -1.574
CC=C C4 -2.306 -2.424 NOT NOT N 0.183
CC=O Na 0.014 -0.595 16.73 -6.87 N -1.005
CC#N P1 0.055 0.540 NOT NOT N -1.574
NC=N P3 1.922 2.106 NOT 12.18 B -7.411
NC=O P3 2.100 2.106 16.67 -0.71 B -2.865
OC=O P1 0.590 0.540 4.27 NOT A -4.695
FC=C C5 -1.400 -1.656 NOT NOT N 0.269
NN=C P1 0.439 0.540 NOT 5.69 B -1.574
ON=C P1 0.426 0.540 10.94 3.14 A -1.574
CC(C)N Nd 0.069 -0.595 NOT 10.43 B -3.985
CC(C)O Nda -0.055 -0.595 17.26 -1.81 B -0.895
CC(C)F C5 -1.469 -1.656 NOT NOT N 0.269
CC(F)F N0 -1.084 -1.009 NOT NOT N -0.191
FC(F)F N0 -1.331 -1.009 NOT NOT N -0.191
"""

# Read the dataset string into a pandas DataFrame
df = pd.read_csv(StringIO(dataset_str), sep=" ", header=None)

# Assign column names
df.columns = ["Col1", "Col2", "Col3", "Col4", "Col5", "Col6", "Col7", "Col8"]

# Extracting features (X) and target variable (Y)
X = df[["Col3", "Col4", "Col5", "Col6", "Col7"]]
y = df["Col8"]


# Split the data into training and testing sets
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Initialize LazyRegressor and fit the models
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

