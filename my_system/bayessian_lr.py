import numpy as np
import pandas as pd

import bambi as bmb
from bambi import Prior


class BayesianLR:
    # def __init__(self, formula:str=None, priors:str=None):
    #     self.formula = formula
    #     self.priors = priors
    
    def fit(self, train:pd.DataFrame):
        # if target is None:
        target = train.columns[-1]
        self.target = target
        features = list(set(train.columns) ^ set([target]))
        # if self.formula is None:
        self.formula = f'{target} ~ {" + ".join(features)}'
        # if self.priors is None:
        self.priors = {col: Prior('Normal', mu=0, sigma=100) for col in features}
        self.model = bmb.Model(self.formula, train, priors=self.priors, )
        self.fitted = self.model.fit(draws=100, tune=100)
    
    def predict(self, X_test: pd.DataFrame):
        self.model.predict(idata=self.fitted, data=X_test)
        return self.fitted.posterior[self.target+'_mean'][0].mean(axis=0).values
    
    def fit_predict(self, train: pd.DataFrame, X_test: pd.DataFrame, target:str=None):
        self.fit(train)
        return self.predict(X_test)