import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import json
import pickle

class Pred():

    def __init__(self, model_name, dataX, dataY, path):
        self.X = dataX
        self.Y = dataY
        self.path = path
        self.model_name = model_name
    
    def choose_regression(self):
        if self.model_name == 'linear_regression' :
            res = self.train_linear_regression()
        elif self.model_name == 'gb' :
            res = self.train_gb_regression()
        elif self.model_name == 'rf' :
            res = self.train_rf_regression()
        
        return res

    def pred_linear_regression(self):
        self.model = pickle.load(open(self.path, 'rb'))
        predictions = self.model.predict(self.X)
        return predictions.squeeze().tolist(), self.Y.FULLVAL.to_numpy()


    def train_gb_regression(self):
        pass

    def train_rf_regression(self):
        pass