import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

# reading the data, giving names to each column
raw_data = pd.read_csv('data/iris/iris.data', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
                                                     'class'])

data = raw_data.copy()

label_encoder = LabelEncoder()  # label encoder converts categorical values (here class name) to numbers (here 0, 1, 2)
data['class'] = label_encoder.fit_transform(data['class'])

predict = 'petal_width'     # chose petal_width as value to predict

X = np.array(data.drop([predict], axis=1))  # dividing all data into X values and Y, which is label values
Y = np.array([data[predict]])

Y = Y.reshape((150, 1))     # reshaping Y array as X.shape[0] and Y.shape[0] must be matching

# dividing X and Y into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

linear = linear_model.LinearRegression()    # creating linear model
linear.fit(x_train, y_train)    # finding the line for train values (training model)

print('Model accuracy:', linear.score(x_test, y_test))  # checking its accuracy by using test set

y_predictions = linear.predict(x_test)    # predicting label by using test set

for i, y_prediction in enumerate(y_predictions):
    print('Prediction:', y_prediction[0])
    print('Label:', y_test[i][0])
    print('Attributes:\n', x_test[i])
    print('*' * 20)



