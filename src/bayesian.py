from bayes_opt import BayesianOptimization
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def NMAE(true : np.array, pred : np.array
    ) -> float:
    mae = np.mean(np.abs(true-pred))
    return -( mae / np.mean(np.abs(true)))

class optimize : 
    def __init__(self, X, y, total_pbounds : dict) : 
        self.data = train_test_split(X, y, test_size = 0.2, shuffle = True)
        self.total_pbounds = total_pbounds
    
    def opt(self, max_depth, n_estimators):
        X_train, X_valid, y_train, y_valid = self.data
        pbounds = {
                    'max_depth' : int(round(max_depth)),
                    'n_estimators' : int(round(n_estimators))
                    }
        model = RandomForestRegressor(**pbounds)
        model.fit(X_train,y_train)
        return NMAE(y_valid, model.predict(X_valid))

    def __call__(self, init_points, n_iters) : 
        BO = BayesianOptimization(
                                    f = self.opt, 
                                    pbounds = self.total_pbounds, 
                                    random_state = 0
                                    )
        BO.maximize(init_points = init_points, n_iter = n_iters) 
        max_params = BO.max['params']
        max_params['max_depth'] = int(max_params['max_depth'])
        max_params['n_estimators'] = int(max_params['n_estimators'])
        return RandomForestRegressor(**max_params)