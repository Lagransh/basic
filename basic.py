import pandas as pd
import numpy as np
import ast

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

        res_array = np.copy(array)

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
    
    

class DateExploder():

    def __init__(self, freq='1D', unit=None, other_str=False, first=True, last=True):
        self.freq = freq
        self.unit = unit
        self.other_str = other_str
        self.first = int(not first)
        self.last = last
    
    
    def _check(self, array):
        if array.__class__ == np.ndarray and array.shape[1] > 1:
            array = pd.DataFrame(x)
        if array.__class__ != pd.DataFrame:
            raise ValueError("Type is not supported")
        
        for col in self.date_cols:
            try:
                array[col] = pd.to_datetime(array[col], unit=self.unit)
            except ValueError:
                raise ValueError("Incorrect date cols")
        return array


    def transform(self, array, from_date=None, to_date=None, other=None):
        self.date_cols = [to_date, from_date]
        if other:
            if isinstance(other, list):
                self.other = other
            else:
                self.other = [other]

        if not from_date or not to_date:
            raise ValueError("Enter the date cols")
        if isinstance(from_date, list) or isinstance(to_date, list):
            raise ValueError("Enter date col, not list")

        res_array = self._check(array).copy()
        if self.last:
            res_array['date_list'] = res_array[self.date_cols].apply(lambda date: pd.date_range(date[1], date[0], freq=self.freq).tolist()[self.first:], axis=1)
        else:
            res_array['date_list'] = res_array[self.date_cols].apply(lambda date: pd.date_range(date[1], date[0], freq=self.freq).tolist()[self.first: -1], axis=1)

        res_array = res_array.drop(self.date_cols, axis=1)

        if not other:
            res_array = res_array.explode('date_list').rename(columns={'date_list':'date'}).reset_index(drop=True)
        else:
            if self.other_str:
                for col in self.other:
                    res_array[col] = res_array[col].apply(lambda x: ast.literal_eval(x) if not isinstance(x, list) else x)
                    
            if res_array['date_list'].apply(lambda x: len(x)).mean() == res_array[self.other].applymap(lambda x: len(x)).mean().mean():
                res_array = res_array.apply(pd.Series.explode).rename(columns={'date_list':'date'}).reset_index(drop=True)
            else:
                res_array['duration'] = res_array['date_list'].apply(lambda x: len(x))
                
                for col in self.other:
                    res_array[col] = res_array[[col] + ['duration']].apply(lambda cols: cols[0] + [np.nan] * (cols[1] - len(cols[0])), axis=1)
                res_array = res_array.drop('duration', axis=1)

                if res_array['date_list'].apply(lambda x: len(x)).mean() == res_array[self.other].applymap(lambda x: len(x)).mean().mean():
                    res_array = res_array.apply(pd.Series.explode).rename(columns={'date_list':'date'}).reset_index(drop=True)
                else:
                    raise ValueError("Other values has higher length then date range")
        
        return res_array
    
    

class GeoSeparation():

    def __init__(self, d0_squares=5, d1_squares=5, d2_squares=None, n_regions=None, dim=2, distance=False):
        self.d_squares = (d0_squares, d1_squares, d2_squares)
        self.n_regions = n_regions
        self.dim = dim
        self.distance = distance
        self.d0_max = None
        self.y = None


    def _check(self, x, y=False):
        if x.__class__ in [pd.DataFrame, pd.Series]:
            x = x.values
        if isinstance(x, list):
            x = np.array([x]) if not y else np.c_[x]
        if x.__class__ == np.ndarray:
            if not y and x.shape[1] != self.dim:
                raise ValueError("X must be {}-dimensional".format(self.dim))
            if y and x.shape[1] > 1:
                raise ValueError("Multi-Y is not supported")
            return x
        else:
            raise ValueError("Type is not supported")


    def fit(self, x, y=None):
        x = self._check(x)
        self.d0_max, self.d0_min = x[:, 0].max(), x[:, 0].min()
        self.d1_max, self.d1_min = x[:, 1].max(), x[:, 1].min()
        self.d2_max, self.d2_min = (x[:, 2].max(), x[:, 2].min()) if self.dim == 3 else None, None
        self.feature_names = ['d0_square','d1_square']
        self.feature_names += ['d2_square'] if self.dim == 3 else []
        self.feature_names += ['square']

        if self.n_regions:
            from sklearn.cluster import KMeans
            from sklearn.neighbors import NearestNeighbors as NN

            self.cluster = KMeans(n_clusters=self.n_regions).fit(x)
            self.feature_names += ['region']

            if self.distance:
              nn = {}
              clusters = np.c_[self.cluster.predict(x)]
              for i in range(self.n_regions):
                  if np.where(clusters == i)[0].shape[0] > 1:
                      self.feature_names += ['nearest_dist_to_' + str(i) + '_region']
                      nn[i] = NN(n_neighbors=2).fit(x[np.where(clusters == i)[0]])
                  else:
                      nn[i] = None
              self.nn = nn

        if y is not None:
            from sklearn.neighbors import NearestNeighbors as NN

            y = self._check(y, y=True)
            if np.unique(y).shape[0] > 15:
                raise ValueError("Y can use only for classification where nunique count less than 15")
            
            ys = {}
            for i, val in enumerate(np.unique(y)):
                if np.where(y == val)[0].shape[0] > 1:
                    self.feature_names += ['nearest_dist_to_' + str(val) + '_y']
                    ys[i] = NN(n_neighbors=2).fit(x[np.where(y == val)[0]])
                else:
                    ys[i] = None
            self.ys = ys
          
        return self


    def transform(self, x):
        x = self._check(x)
        if not self.d0_max:
            raise ValueError("Didn't fit yet")

        d0_len = self.d0_max - self.d0_min
        d1_len = self.d1_max - self.d1_min
        d2_len = (self.d2_max - self.d2_min) if self.dim == 3 else None

        res_array = np.copy(x)
        res_array[:, 0] = (x[:, 0] - self.d0_min) // (d0_len / self.d_squares[0])
        res_array[:, 1] = (x[:, 1] - self.d1_min) // (d1_len / self.d_squares[1])
        if self.dim == 3:
            res_array[:, 2] = (x[:, 2] - self.d2_min) // (d2_len / self.d_squares[2])
            square = res_array[:, [0]] * self.d_squares[1] + res_array[:, [1]] * self.d_squares[2] + res_array[:, [2]]
        else:
            square = res_array[:, [0]] * self.d_squares[1] + res_array[:, [1]]

        res_array = np.concatenate([res_array, square], axis=1)

        if self.n_regions:
            res_array = np.concatenate([res_array, np.c_[self.cluster.predict(x)]], axis=1)

            if self.distance:
                for key in self.nn.keys():
                    if self.nn[key]:
                        res_array = np.concatenate([res_array, self.nn[key].kneighbors(x)[0][:, [1]]], axis=1)
                
        if self.ys:
            for key in self.ys.keys():
                if self.ys[key]:
                    res_array = np.concatenate([res_array, self.ys[key].kneighbors(x)[0][:, [1]]], axis=1)
        
        return res_array


    def fit_transform(self, x, y=None):
        self = self.fit(x, y=y)
        return self.transform(x)

    
    def get_features_names(self):
        return self.feature_names
