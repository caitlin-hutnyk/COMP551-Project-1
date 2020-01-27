import numpy as np

ERROR_VALUE = -999
filename = 'data/ionosphere.data'

# converts 'g' and 'b' into 1 and 0 respectively
def convert(a):
    if a == b'g':
        return 1.0
    elif a == b'b':
        return 0.0
    else:
        return ERROR_VALUE


# read data from file with data type float, split on ',' and convert the final column to a numeric value
data = np.genfromtxt(filename, dtype='<f8', delimiter=',', converters={34: convert})
# split the final column to be label vector
y = np.array((data[:, 34]))
# delete the label vector from the dataset to leave the design matrix
x = np.delete(data, 34, 1)
# ensure arrays have been generated correctly
assert np.array_equal(data[0, :], np.append(x[0, :], y[0])), "Error in reading data or converting"
