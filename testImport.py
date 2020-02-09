import numpy as np
import pandas as pd
import constants

def read_data(which, type_):
	np.set_printoptions(threshold=np.inf)
	pd.set_option('display.max_columns', 500)

	if which = constants.IONOSPHERE:
		filename = 'data/ionosphere.data'
		columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
		         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 'assessment']
		columns = list(map(str, columns))

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


	elif which = constants.CENSUS:
		# trainfile = 'data/ionosphere.data'
		trainfile = 'data/adult.data'
		testfile = 'data/adult.test'
		columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
		            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
		           'salary']

		# import txt file, keep original layout and format, and display top five instances as sample
		data = pd.read_csv(trainfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
		test = pd.read_csv(testfile, sep=",", header=None, na_values=[" ?", "?", "99999"," 99999"], names=columns)
		data.head()
		test.head()
		data.info()
		test.info()

		data = data.dropna(axis='index', how="any")
		test = test.dropna(axis='index', how='any')

		# convert categorical variables into dummy variables
		data_new = pd.get_dummies(data)
		data_new.head()

		test_new = pd.get_dummies(test)
		test_new.head()

		# data_missing = data[data_new.isnull().any(axis=1)].head()
		# print(data_missing)  # no missing is found

		# convert from panda data frame to numpy arrays
		data_new_arr = pd.DataFrame(data_new).to_numpy()
		test_new_arr = pd.DataFrame(test_new).to_numpy()

		# split original matrix into x and y
		train_y = np.array(data_new_arr[:,-2:-1])
		train_x_con = np.array(data_new_arr[:, :6])
		train_x_cat = np.array(data_new_arr[:, 6:-2])

		test_y = np.array(test_new_arr[:,-2:-1])
		test_x_con = np.array(test_new_arr[:, :6])
		test_x_cat = np.array(test_new_arr[:, 6:-2])

		if type_:
			return np.array(data_new_arr[:,-2]), train_y, np.array(test_new_arr[:,-2]), test_y
		return train_x_con, train_x_cat, train_y, test_x_con, test_x_cat, test_y

	elif which = constants.POKER:

	elif which = constants.CREDIT: