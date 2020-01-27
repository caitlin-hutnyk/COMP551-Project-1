import numpy as np
import pandas as pd

trainfile = 'data/ionosphere.data'

# import txt file, keep original layout and format, and display top five instances as sample
data = pd.read_csv(trainfile, sep=",", header=None)
data.head()
data.info()

# convert categorical variables into dummy variables
data_new = pd.get_dummies(data, columns=[34])
data_new.head()

# show instances that contain missing features
data_missing = data[data_new.isnull().any(axis=1)].head()
print(data_missing)  # no missing is found

# show basic stats of features and y
print(data_new.describe())

# plot histograms for features
import matplotlib.pyplot as plt
data_new.hist(bins=50, figsize=(20,15))
plt.show()

# split original matrix into x and y
data_arr = pd.DataFrame(data_new).to_numpy()
y = np.array(data_arr[: ,-1])
x = np.array(data_arr[:33])

