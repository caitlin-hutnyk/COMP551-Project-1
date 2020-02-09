import numpy as np
import pandas as pd
import constants
from sklearn.preprocessing import normalize

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 500)

def read_data(which, type_):

	if which = constants.IONOSPHERE:
		trainfile = 'data/ionosphere.data'
		columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:,-2:-1])
		train_x_con = np.array(data_new_arr[:, :-2])
		train_x_cat = -1


	elif which = constants.CENSUS:
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

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:,-2:-1])
		train_x_con = np.array(data_new_arr[:, :6])
		train_x_cat = np.array(data_new_arr[:, 6:-2])



		if type_:
			return np.array(data_new_arr[:,-2]), train_y, np.array(test_new_arr[:,-2]), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

	elif which = constants.POKER:
		trainfile = 'data/poker-hand-training-true.data'
		testfile = 'data/poker-hand-testing.data'
		columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999", " 99999"], names=columns)

		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data.astype(str))
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:, -10:])
		train_x_con = -1
		train_x_cat = np.array(data_new_arr[:,:-10])

		if type_:
			return np.array(data_new_arr[:, -2]), train_y, np.array(test_new_arr[:, -2]), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y


	elif which = constants.CREDIT:

		trainfile = 'data/crx.data'
		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999", " 99999"])
		data.head()
		data.info()

		data = data.dropna(axis='index', how="any")

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:, -2:-1])
		train_x_con = np.array(data_new_arr[:, :6])
		train_x_cat = np.array(data_new_arr[:, 6:-2])

		if type_:
			return np.array(data_new_arr[:, -2]), train_y, np.array(test_new_arr[:, -2]), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y