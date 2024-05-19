import pandas as pd

import bamt.preprocessors as pp
from bamt.networks import ContinuousBN
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder


class LinearRegression:
    def __init__(self):
        pass

    def fit(self, train: pd.DataFrame, target:str =None):
        if target is None:
            target = train.columns[-1]
        # Определяем ребра
        edges = [(col, target) for col in train.columns[:-1]]
        # Создаем энкодер и discretizer
        encoder = LabelEncoder()
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal")
        p = pp.Preprocessor([("encoder", encoder), ("discretizer", discretizer)])
        p.apply(train)
        # Создаем Байесовскую сеть и передаем параметры
        self.bn = ContinuousBN(use_mixture=True)
        self.bn.add_nodes(p.info)
        self.bn.set_structure(edges=edges)
        # Обучаем и предсказываем
        self.bn.fit_parameters(train)
        self.bn.plot('foo.html')
    
    def predict(self, X_test: pd.DataFrame, target):
        pred = self.bn.predict(test=X_test)
        return pred[target]
    
    def fit_predict(self, train: pd.DataFrame, X_test: pd.DataFrame, target:str =None):
        if target is None:
            target = train.columns[-1]
        self.fit(train, target)
        return self.predict(X_test, target)
    