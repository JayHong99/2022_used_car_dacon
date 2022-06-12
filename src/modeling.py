from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

from src.bayesian import optimize
from src.shap import shap_explain



class RF : 
    def __init__(self, train : pd.DataFrame, test : pd.DataFrame, pbounds : dict) :  
        self.X = train.drop(columns = ['target'])
        self.feature_names = list(self.X.columns)
        self.X = self.X.to_numpy()
        self.y = train[['target']].to_numpy().reshape(-1)
        self.test = test.to_numpy()
        self.pbounds = pbounds
    
    def get_best_model(self, init_points, n_iters) -> None : 
        opti = optimize(self.X, self.y, self.pbounds)
        self.model = opti(init_points, n_iters)
        self.model.fit(self.X, self.y)
    
    def predict(self) : 
        return self.model.predict(self.test)

    def explain(self) : 
        shap_explain(self.model, self.X, self.feature_names)