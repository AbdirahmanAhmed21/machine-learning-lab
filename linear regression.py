import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
#
data = fetch_california_housing()

#convert to pandas dataframe for better handling
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#inpect data(tge first few rows of dataset)

print(X.head())

X.single_feature = X['MedInc']
y_target = y

X_train, X_text, y_train, y_test =  train_test_split(X.single_feature, y_target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


