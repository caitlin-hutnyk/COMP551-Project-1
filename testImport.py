import numpy as np

filename = 'ionosphere.data'
data = np.genfromtxt(filename, delimiter=',')

print(data.shape)
