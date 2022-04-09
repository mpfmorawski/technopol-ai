import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

class Train:
    "Pipeline for train model"

    def __init__(self, dataX, dataY):
        #self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = self.prepare_test_train_data(dataX, dataY)
        self.X = dataX
        self.Y = dataY
        self.model = None
        self.kf = KFold(n_splits=10, random_state=7, shuffle=True)
    
    def prepare_test_train_data(self, dataX, dataY):
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(dataX, dataY, test_size=0.2, random_state=42)
        return Xtrain, Ytrain, Xtest, Ytest 

    def evaluate_linear_regression(self):
        self.model = LinearRegression()
        # print(f"Model coefficient: {self.model.coef_}, Model intercept: {self.model.intercept}")
        # return self.model

        results_mae = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_absolute_error')
        print("MAE: %.3f (%.3f)" % (results_mae.mean(), results_mae.std()))

        self.model = LinearRegression()

        results_mse = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_squared_error')
        print("MSE: %.3f (%.3f)" % (results_mse.mean(), results_mse.std()))

        self.model = LinearRegression()

        results_r2 = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='r2')
        print("R2: %.3f (%.3f)" % (results_r2.mean(), results_r2.std())) 






    

