import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json

class Train:
    "Pipeline for train model"

    def __init__(self, dataX, dataY, model_name):
        #self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = self.prepare_test_train_data(dataX, dataY)
        self.X = dataX
        self.Y = dataY
        self.model = None
        self.model_name = model_name
        self.kf = KFold(n_splits=10, random_state=7, shuffle=True)
    
    def prepare_test_train_data(self, dataX, dataY):
        Xtrain, Ytrain, Xtest, Ytest = train_test_split(dataX, dataY, test_size=0.2, random_state=42)
        return Xtrain, Ytrain, Xtest, Ytest 

    def evaluate_regression(self):
        if self.model_name == 'linear_regression' :
            res = self.evaluate_linear_regression()
        elif self.model_name == 'gb' :
            res = self.evaluate_gb_regression()
        elif self.model_name == 'rf' :
            res = self.evaluate_rf_regression()
        
        return res

    def evaluate_linear_regression(self):
        self.model = LinearRegression()
        # print(f"Model coefficient: {self.model.coef_}, Model intercept: {self.model.intercept}")
        # return self.model

        results_mae = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_absolute_error')
        print("MAE: %.3f (%.3f)" % (results_mae.mean(), results_mae.std()))
        mae_ref = results_mae.mean()/self.Y.mean()
        print(f"MAE_ref: {mae_ref}")
        self.model = LinearRegression()

        results_mse = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_squared_error')
        print("MSE: %.3f (%.3f)" % (results_mse.mean(), results_mse.std()))

        self.model = LinearRegression()

        results_r2 = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='r2')
        print("R2: %.3f (%.3f)" % (results_r2.mean(), results_r2.std())) 

        print(type(results_mae))
        rm = results_mae.tolist()
        res = {
            "len" : str(len(self.X)),
            "mean_y" : str(self.Y.mean()),
            "results" : str(rm),
            "MAE_mean" : str(results_mae.mean()),
            "MAE_std" : str(results_mae.std()),
            "MAE_mean_ref" : str( results_mae.mean()/self.Y.mean() ),
            "R2_mean" : str( results_r2.mean() ),
            "R2_std" : str ( results_r2.std() )
        }

        return res
    
    def evaluate_gb_regression(self):
        self.model = GradientBoostingRegressor(random_state=0)
        # print(f"Model coefficient: {self.model.coef_}, Model intercept: {self.model.intercept}")
        # return self.model

        print(self.Y.mean())

        results_mae = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_absolute_error')
        print("MAE: %.3f (%.3f)" % (results_mae.mean(), results_mae.std()))
        mae_ref = results_mae.mean()/self.Y.mean()
        print(f"MAE_ref: {mae_ref}")

        self.model = GradientBoostingRegressor(random_state=0)

        results_mse = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_squared_error')
        print("MSE: %.3f (%.3f)" % (results_mse.mean(), results_mse.std()))

        self.model = GradientBoostingRegressor(random_state=0)

        results_r2 = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='r2')
        print("R2: %.3f (%.3f)" % (results_r2.mean(), results_r2.std())) 
        rm = results_mae.tolist()
        res = {
            "len" : len(self.X),
            "mean_y" : self.Y.mean(),
            "results" : rm,
            "MAE_mean" : float(results_mae.mean()),
            "MAE_std" : float(results_mae.std()),
            "MAE_mean_ref" : float( results_mae.mean()/self.Y.mean() ),
            "R2_mean" : float( results_r2.mean() ),
            "R2_std" : float ( results_r2.std() ),
        }

        return res

    def evaluate_rf_regression(self):
        self.model = RandomForestRegressor(criterion='absolute_error')
        # print(f"Model coefficient: {self.model.coef_}, Model intercept: {self.model.intercept}")
        # return self.model

        print(self.Y.mean())

        results_mae = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_absolute_error')
        print("MAE: %.3f (%.3f)" % (results_mae.mean(), results_mae.std()))
        mae_ref = results_mae.mean()/self.Y.mean()
        print(f"MAE_ref: {mae_ref}")

        self.model = RandomForestRegressor(criterion='absolute_error')

        results_mse = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='neg_mean_squared_error')
        print("MSE: %.3f (%.3f)" % (results_mse.mean(), results_mse.std()))

        self.model = RandomForestRegressor(criterion='absolute_error')

        results_r2 = cross_val_score(self.model, self.X, self.Y, cv=self.kf, scoring='r2')
        print("R2: %.3f (%.3f)" % (results_r2.mean(), results_r2.std())) 
        rm = results_mae.tolist()
        res = {
            "len" : len(self.X),
            "mean_y" : self.Y.mean(),
            "results" : rm,
            "MAE_mean" : float(results_mae.mean()),
            "MAE_std" : float(results_mae.std()),
            "MAE_mean_ref" : float( results_mae.mean()/self.Y.mean() ),
            "R2_mean" : float( results_r2.mean() ),
            "R2_std" : float ( results_r2.std() ),
        }

        return res






    

