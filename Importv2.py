import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


# will allow dataset to be specified by user later
# dataset 1: ionosphere data, uncomment to use
filename = 'data/ionosphere.data'
columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 'assessment']
columns = list(map(str, columns))

# dataset 2: census data, uncomment to use
filename = 'data/adult.data'
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'salary']

# type_ == 1 if logregression
# type_ == 0 if nb
def read_data(type_):
    print("Reading in {}".format(filename))
    # read in data and replace unknown values with NaN to be easily removed
    data = pd.read_csv(filename, sep=",", na_values=[" ?", "?"], names=columns)
    old_instances = data.shape[0]

    # remove all instances with missing data
    data = data.dropna(axis=0, how="any")
    instances = data.shape[0]

    print("Rows dropped: %d" % (old_instances - instances))
    print(data.shape)

    # create separate y label array and encode as 0/1
    y_label = data.values[:, -1]
    print("Y: {}".format(y_label[0]))
    label_encoder = LabelEncoder()
    y_label = label_encoder.fit_transform(y_label)
    y_label = y_label.reshape(y_label.shape[0], 1)
    print("is Y: {}".format(y_label[0]))

    # remove y label from data
    data = data.iloc[:, :-1]

    # identify categorical and continuous features
    categorical_feature_mask = []
    categorical_feature_mask = (data.dtypes == object)
    categorical_cols = data.columns[categorical_feature_mask].tolist()
    continuous_cols = data.columns[~categorical_feature_mask].tolist()

    # identify index of each columns
    categorical_cols_index = []
    continuous_cols_index = []
    index = 0
    for col in categorical_feature_mask:
        if col is True:
            categorical_cols_index.append(index)
        else:
            continuous_cols_index.append(index)
        index = index + 1

    # create one col for data_continuous so that we can append further cols
    data_continuous = np.ones((data.shape[0], 1))

    print("data continuous instant dtype {}".format(data_continuous.dtype))
    print(data_continuous.shape)

    for i in continuous_cols_index:
        new_col = data.iloc[:, i]
        col_reshape = new_col.values.reshape((data.shape[0], 1))
        data_continuous = np.append(data_continuous, col_reshape, axis=1)

    # drop the first col
    data_continuous = data_continuous[:, 1:]

    # here we have all our continuous vars in one array, so we can drop them
    ct = ColumnTransformer([('dummy_col', OneHotEncoder(sparse=False), categorical_cols)], remainder='passthrough')
    data_categorical = ct.fit_transform(data)
    data_categorical = data_categorical.astype(int)

    # we now have our data separated as continuous and categorical

    # split into training and testing set by 80:20 split
    mask = np.random.rand(data_categorical.shape[0]) < 0.8
    train_validate_categorical = data_categorical[mask]
    train_validate_continuous = data_continuous[mask]
    train_val_y = y_label[mask]

    # final test data
    test_categorical = data_categorical[~mask]
    test_continuous = data_continuous[~mask]
    test_y = y_label[~mask]

    # ensure split is accurate
    assert (train_validate_categorical.shape[0] + test_categorical.shape[0] == instances)

    # putting back together to test --
    X_train = np.concatenate((train_validate_categorical, train_validate_continuous), axis=1)
    X_test = np.concatenate((test_categorical, test_continuous), axis=1)

    # different return types for logregression and bayes
    if type_:
    	return X_train, train_val_y, X_test, test_y
    return train_validate_continuous, train_validate_categorical, train_val_y, test_continuous, test_categorical, test_y
