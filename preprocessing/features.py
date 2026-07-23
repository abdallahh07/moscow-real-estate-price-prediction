import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        X = x.copy()
        X = pd.get_dummies(X, columns=self.variables, drop_first=True)
        return X