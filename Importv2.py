import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# will allow dataset to be specified by user later
# dataset 1: ionosphere data, uncomment to use
filename = 'data/ionosphere.data'
columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 'assessment']
columns = list(map(str, columns))

# dataset 2: census data, uncomment to use
# filename = 'data/adult.data'
# columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
#            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
#            'salary']

# read in data and replace unknown values with NaN to be easily removed
data = pd.read_csv(filename, sep=",", na_values=[" ?", "?"], names=columns)
old_instances = data.shape[0]
print(old_instances)

# remove all instances with missing data
data = data.dropna(axis=0, how="any")
instances = data.shape[0]

print("Rows dropped: %d" % (old_instances - instances))
print(data.shape)
print("Original row:")
# to test any row, change the var
print(data.iloc[0])

# convert categorical values to binary
categorical_feature_mask = (data.dtypes == object)
print(categorical_feature_mask)
categorical_cols = data.columns[categorical_feature_mask].tolist()

print("Categorical variables in columns:")
print(categorical_cols)

ct = ColumnTransformer([('a', OneHotEncoder(sparse=False), categorical_cols)], remainder='passthrough')
data = ct.fit_transform(data)

np.set_printoptions(precision=5)
# to test, can alter the x var
print(np.array(data[0, :]))
print(data.shape)

# split into training and testing set by 80:20 split
mask = np.random.rand(data.shape[0]) < 0.8
train_validate = data[mask]
test = data[~mask]

# ensure split is accurate
assert (train_validate.shape[0] + test.shape[0] == instances)
print("training set: {0} testing set: {1} ".format(train_validate.shape[0], test.shape[0]))

# split into training and validation set by 80:20 split
mask = np.random.rand(train_validate.shape[0]) < 0.8
train = train_validate[mask]
validation = train_validate[~mask]

# ensure split is accurate
assert (train.shape[0] + validation.shape[0] == train_validate.shape[0])
print("training set: {0} validation set: {1} ".format(train.shape[0], validation.shape[0]))
