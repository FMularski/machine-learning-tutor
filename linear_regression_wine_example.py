import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style

raw_data = pd.read_csv('data/wine/winequality-red.csv', sep=';')
data = raw_data.copy()

label = 'quality'

X = np.array(data.drop([label], axis=1))
Y = np.array(data[label])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear_model = linear_model.LinearRegression()
linear_model.fit(x_train, y_train)

accuracy = linear_model.score(x_test, y_test)
print(accuracy)

quality_predictions = linear_model.predict(x_test)

for i, prediction in enumerate(quality_predictions):
    print('Quality [prediction]:', prediction)
    print('Quality [actual]:', y_test[i])
    print()



