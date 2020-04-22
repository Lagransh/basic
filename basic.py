import pandas as pd
import numpy as np

class PolynomialFeatures():

    def __init__(self, degree=1, interactions=True, multi_degree=False, exp=False, obj_func=None):
        self.degree = degree
        self.interactions = interactions
        self.obj_func = obj_func
        self.exp = exp
        self.multi_degree = multi_degree
        self.n_features = None


    def _check(self, x):
        if x.__class__ in [pd.DataFrame, pd.Series]:
            x = x.values
        if isinstance(x, list):
            x = np.array([x])
        if x.__class__ == np.ndarray:
            return x
        else:
            raise ValueError("Type is not supported")


    def fit(self, x, y=None):
        x = self._check(x)
        num_cols = []
        cat_cols = []
        for col in range(x.shape[1]):
            try:
                x[:, col].astype(np.number)
                num_cols.append(col)
            except ValueError:
                cat_cols.append(col)
        self.n_features = x.shape[1]
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        if self.obj_func:
            self.preprocess(x[:, cat_cols])
        else:
            self.obj_model = None
        return self


    def preprocess(self, array):
          if self.obj_func == 'one_hot':
              from sklearn.preprocessing import OneHotEncoder as OHE
              self.obj_model = OHE(handle_unknown='ignore', sparse=False).fit(array)
          return self

    
    def transform(self, x):
        if not self.n_features:
            raise ValueError("Is not trained")

        x = self._check(x)
        array = x[:, self.num_cols]
        if x.shape[1] != self.n_features:
            raise ValueError("X shape does not match training shape")

        res_array = array

        for deg in range(2, self.degree + 1):
            res_array = np.concatenate([res_array, array ** deg], axis=1)

        for mult_col in range(2 * self.degree):
            for loc in range(0, self.degree - mult_col - 1):
                res_array = np.concatenate([res_array, res_array[:, [mult_col]] * res_array[:, mult_col + array.shape[1] * loc + 1: mult_col + array.shape[1] * (loc + 1)]], axis=1)

        if self.interactions == False:
            res_array = np.concatenate([res_array[:, :array.shape[1]], res_array[:, array.shape[1] * self.degree:]], axis=1)
        
        if self.multi_degree:
            multi_array = array
            for mult_col in range(array.shape[1]):
                if array[:, mult_col].min() > 0.001 and array[:, mult_col].max() < 32:
                    other_cols = [e for e in range(array.shape[1])]
                    other_cols.remove(mult_col)
                    multi_array = np.concatenate([multi_array, array[:, other_cols] ** array[:, [mult_col]]], axis=1)
            res_array = np.concatenate([res_array, multi_array[:, array.shape[1]:]], axis=1)
        
        if self.exp:
            res_array = np.concatenate([res_array, np.log(array.astype(np.float32)), np.exp(array.astype(np.float32))], axis=1)

        if self.obj_model:
            res_array = np.concatenate([res_array, self.obj_model.transform(x[:, self.cat_cols])], axis=1)

        return res_array


    def fit_transform(self, x):
        self = self.fit(x)
        return self.transform(x)
