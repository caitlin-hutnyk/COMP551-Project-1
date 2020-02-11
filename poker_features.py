import numpy as np 

# x is Nx10 (n number of training examples)
# convert card data to new base data
# with 0-9 being cards...
# 10 is flush
# 11 is straight
# 12 is quads
# 13 is trips
# 14 is pair
def convert(x):
	result = np.c_[x, np.zeros((N,5))]
	for i in result:
		# flush!
		if i[0] == i[2] and i[0] == i[4] and i[0] == i[6] and i[0] == i[8]:
			i[10] = 1 

		# straight!
		ranks = []
		for j in range(1, 10, 2):
			ranks.append(i[j])
		ranks.sort()
		flush = True
		for j in range(len(ranks) - 2):
			if ranks[j] + 1 != ranks[j+1]:
				flush = False
				break
		if flush:
			if ranks[3] + 1 == ranks[4] or ranks[3] == 13 and ranks[4] == 1:
				i[11] = 1

		# quads!
		quads = True
		for j in range(4):
			if ranks[j] != ranks[j+1]:
				quads = False
				break
		if quads:
			i[12] = 1
			i[13] = 1
			i[14] = 1
			continue
		else:
			quads = True
			for j in range(1,5):
				if ranks[j] != ranks[j+1]:
					quads = False
					break
		if quads:
			i[12] = 1
			i[13] = 1
			i[14] = 1
			continue

		# trips!
		for j in range(3):
			trip = True
			for k in range(j,2+j):
				if ranks[k] != ranks[k+1]:
					trip = False
					break
			if trip:
				i[13] = 1
				i[14] = 1
				break

		# pair
		if i[14]:
			continue
		for j in range(4):
			pair = True
			for k in range(j,3+j):
				if ranks[k] != ranks[k+1]:
					pair = False
					break
			if pair:
				i[14] = 1
				break

	return result
