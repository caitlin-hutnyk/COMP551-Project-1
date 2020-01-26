
import numpy as np

filename = 'data/ionosphere.data'
data = np.genfromtxt(filename, delimiter=',')

print(data.shape)


