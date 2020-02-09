import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 500)

# trainfile = 'data/ionosphere.data'
trainfile = 'data/adult.data'

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'salary']

# import txt file, keep original layout and format, and display top five instances as sample
data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
data.head()
data.info()

data = data.dropna(axis='index', how="any")

# convert categorical variables into dummy variables
data_new = pd.get_dummies(data)
data_new.head()

# data_missing = data[data_new.isnull().any(axis=1)].head()
# print(data_missing)  # no missing is found

# convert from panda data frame to numpy arrays
data_new_arr = pd.DataFrame(data_new).to_numpy()

# split original matrix into x and y
train_y = np.array(data_new_arr[:,-2:-1])
train_x_con = np.array(data_new_arr[:, :6])
train_x_cat = np.array(data_new_arr[:, 6:-2])