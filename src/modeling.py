from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import numpy as np

def NMAE(true : np.array, pred : np.array
    ) -> float:
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score

class modeling : 
    def __init__(self, X_train : np.array, X_test : np.array, y_train : np.array, y_test : np.array) : 
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train(self, model) : 
        model.fit(self.X_train, self.y_train)
    
    def valid(self, model) -> np.array : 
        model_name = model.__class__.__name__
        pred = model.predict(self.X_test)
        pred[pred < 0] = 0
        if self.y_test is not None : 
            score = self.score(pred)
            print(f'{model_name} VALIDATION SCORE : {score:.4f}')
        return pred
    
    def score(self, pred) : 
        score = NMAE(self.y_test, pred)
        return score


    def linear_regression(self) : 
        model = LinearRegression()
        self.train(model)
        return self.valid(model)

    def rf(self) : 
        model = RandomForestRegressor()
        self.train(model)
        return self.valid(model)
    
    def ext(self) : 
        model = ExtraTreesRegressor()
        self.train(model)
        return self.valid(model)

    def lgbm(self) : 
        model = LGBMRegressor()
        self.train(model)
        return self.valid(model)
    
    def xgb(self) : 
        model = XGBRegressor()
        self.train(model)
        return self.valid(model)