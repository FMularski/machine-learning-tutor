import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

raw_data = pd.read_csv('data/student/student-mat.csv', sep=';')
data = raw_data[['G1', 'G2', 'G2', 'studytime', 'failures', 'absences']]

print(data.head(10))
