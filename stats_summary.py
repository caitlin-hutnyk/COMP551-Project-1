import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 500)

# dataset1
# import file and print samles entries and basic info
trainfile1 = 'data/ionosphere.data'
data1 = pd.read_csv(trainfile1, sep=",", header=None)
data1.head()
data1.info()

# check if dataset has any nan values, print nan entries
data1_missing = data1[data1.isna().any(axis=1)]
print (data1_missing)

data1_new = data1.dropna(axis=0, how="any")
print(data1_new)

print(data1_new.describe()) # show stats summary

# plot histograms for features

sns.set(style="ticks", color_codes=True)

data1_new.hist(bins=50, figsize=(20,15))
sns.catplot(x=34, kind="count", palette="ch:.25", data=data1_new)
# the correlations are different between the pair of features, some shows linear correlation some just
# show random distributed data points
plt.scatter(data1_new [16], data1_new[14]) #we can see feature 14 and 16 has linear correlation
plt.scatter(data1_new [25], data1_new[6]) # for feature 6 and 25 it's hard to find correlation, but we can see most of data points of feature fall under greater values
plt.scatter(data1_new [34], data1_new[6])
plt.scatter(data1_new [34], data1_new[16]) # by looking at the distribution of features under different class, we can see their pattern are quite different, therefore we can say this feature has strong predicting ability
plt.show()
data1_new[34].value_counts()



# dataset2
trainfile2 = 'data/adult.data'
columns2 = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'salary']
data2 = pd.read_csv(trainfile2, sep=",", na_values=[" ?", "?"," 99999"], header=None, names=columns2)
data2_missing = data2[data2.isna().any(axis=1)]
print(data2_missing)
data2_new = data2.dropna(axis=0, how="any")
print(data2_new)

print(data2_new.describe())

sns.set(style="ticks", color_codes=True)
data2_new.hist(bins=50, figsize=(20,15))
sns.catplot(x="salary", kind="count", palette="ch:.25", data=data2_new)
plt.scatter(data2_new['salary'], data2_new['age'], color='r') # age range of salary class <=50k is wider, most of 20
sns.scatterplot(x="age", y="fnlwgt", hue="sex", data=data2_new)
plt.show()
data2_new["salary"].value_counts()

sns.catplot(x="sex", y="education-num", kind="swarm", data=data2_new)


# dataset3
trainfile3 = 'data/poker-hand-training-true.data'
columns3 = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS']
data3 = pd.read_csv(trainfile3, sep=",", header=None, names=columns3)
data3_missing = data3[data3.isna().any(axis=1)]
print(data3_missing)
data3_new = data3.dropna(axis=0, how="any")
print(data3_new)

print(data3_new.describe())

sns.set(style="ticks", color_codes=True)
data3_new.hist(bins=50, figsize=(20,15))
sns.catplot(x="CLASS", kind="count", palette="ch:.25", data=data3_new)
plt.show()
data3_new["CLASS"].value_counts()

# dataset4
trainfile4 = 'data/crx.data'
data4 = pd.read_csv(trainfile4, sep=",", na_values=[" ?", "?"], header=None)
print(data4.shape)
data4_missing = data4[data4.isna().any(axis=1)]
print(data4_missing)
print(data4_missing.shape)
data4_new = data4.dropna(axis=0, how="any")
print(data4_new)
print(data4_new.shape)

print(data4_new.describe())

sns.set(style="ticks", color_codes=True)
data4_new.hist(bins=50, figsize=(20,15))
sns.scatterplot(x=1, y=2, data=data4_new)
sns.scatterplot(x=15, y=7, data=data4_new)
plt.show()
data4_new[16].counts()

