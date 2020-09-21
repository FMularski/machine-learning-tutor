import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style

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

linear = linear_model.LinearRegression()    # the model
linear.fit(x_train, y_train)    # fitting the line for train set
accuracy = linear.score(x_test, y_test)     # score() returns the accuracy of the model based on running test data

print('Coefficient:\n', linear.coef_)       # coefficient is a in y = ax + b, one for each attribute
# 5 attributes so 5-dim space, the bigger coefficient, the greater influence the given attribute has on the result
print('Intercept:\n', linear.intercept_)    # intercept is b in y = ax + b

# PREDICTING STUDENT'S GRADE

y_predictions = linear.predict(x_test)      # predicts label based on testing set
for i in range(len(y_predictions)):         # iterating over all predictions
    print(y_predictions[i], x_test[i], y_test[i])   # displaying predictions, attributes and actual label value

p = 'G1'
style.use('ggplot')
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel('Final Grade')
plt.show()



