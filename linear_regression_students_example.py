import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

raw_data = pd.read_csv('data/student/student-mat.csv', sep=';')
data = raw_data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]    # select specific columns

# print(data.head(10))    # print top rows, 5 by default

predict = 'G3'  # aka 'label', predicted based on the attributes (other columns)

X = np.array(data.drop([predict], axis=1))  # array with attributes (without G3), axis means drop by columns
# print(X)
Y = np.array(data[predict])  # array with label
# print(Y)

# now when got X and Y, need to divide these into train and test sets
# we can set the proportion of train and test sets, here 0.9 train and 0.1 test
# cannot set 100% as test because the algorithm will know all the answers before testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

