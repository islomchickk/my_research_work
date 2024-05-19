import pandas as pd

from catboost import CatBoostRegressor


class CatboostRegressor:
    def __init__(self):
        self.cat = CatBoostRegressor(verbose=500, loss_function='RMSE')
        
    def fit(self, train:pd.DataFrame, target:str =None):
        if target is None:
            target = train.columns[-1]
        X_train, y_train = train.drop(target, axis=1), train[target]
        self.cat = self.cat.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.cat.predict(X_test)
    
    def fit_predict(self, train, X_test, target:str =None):
        if target is None:
            target = train.columns[-1]
        self.fit(train=train, target=target)
        return self.predict(X_test=X_test)