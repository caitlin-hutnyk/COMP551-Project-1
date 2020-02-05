import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# will allow dataset to be specified by user later
# dataset 1: ionosphere data, uncomment to use
# filename = 'data/ionosphere.data'
# columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
 #          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 'assessment']
# columns = list(map(str, columns))

# dataset 2: census data, uncomment to use
filename = 'data/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'salary']

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

categorical_feature_mask = []
# convert categorical values to binary
categorical_feature_mask = (data.dtypes == object)
print(categorical_feature_mask)
categorical_cols = data.columns[categorical_feature_mask].tolist()
continuous_cols = data.columns[~categorical_feature_mask].tolist()

print(categorical_cols)

categorical_cols_index = []
continuous_cols_index = []
index = 0

for col in categorical_feature_mask:
    if col is True:
        categorical_cols_index.append(index)
    else:
        continuous_cols_index.append(index)
    index = index + 1

data_continuous = np.ones((instances, 1))
for i in continuous_cols_index:
    print("i is {} ".format(i))
    new_col = data.values[:, i]
    col_reshape = new_col.reshape((data.shape[0], 1))
    data_continuous = np.append(data_continuous, col_reshape, axis=1)

data_continuous = data_continuous[:, 1:]
print(data_continuous.shape)

# here we have all our continuous vars in one array, and we drop them in the column transformer
print("continuous data array shape:")
print(data_continuous.shape)
ct = ColumnTransformer([('a', OneHotEncoder(sparse=False), categorical_cols)], remainder='passthrough')
data_categorical = ct.fit_transform(data)

print(data_categorical)


np.set_printoptions(precision=5)
print("original data row 0:")
print(data.iloc[0])
print("continuous columns")
print(data_continuous[0])


# split into training and testing set by 80:20 split
mask = np.random.rand(data_categorical.shape[0]) < 0.8
train_validate_categorical = data_categorical[mask]
train_validate_continuous = data_continuous[mask]

# final
test_categorical = data_categorical[~mask]
test_continuous = data_continuous[~mask]

# ensure split is accurate
assert (train_validate_categorical.shape[0] + test_categorical.shape[0] == instances)

print("training set: {0} testing set: {1} ".format(train_validate_categorical.shape[0], test_categorical.shape[0]))

# split into training and validation set by 80:20 split
mask = np.random.rand(train_validate_categorical.shape[0]) < 0.8

# final
train_categorical = train_validate_categorical[mask]
validation_categorical = train_validate_categorical[~mask]
train_continuous = train_validate_continuous[mask]
validation_continuous = train_validate_continuous[~mask]

# ensure split is accurate
assert (train_categorical.shape[0] + validation_categorical.shape[0] == train_validate_categorical.shape[0])
print("training set: {0} validation set: {1} ".format(train_categorical.shape[0], validation_categorical.shape[0]))
