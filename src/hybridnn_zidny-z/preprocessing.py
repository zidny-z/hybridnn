import numpy as np

class MinMaxScaler:
     def __init__(self):
          self.min_val = None
          self.max_val = None

     def get_min_val(self):
          return self.min_val
     
     def get_max_val(self):
          return self.max_val

     def fit(self, data):
          self.min_val = np.min(data, axis=0)
          self.max_val = np.max(data, axis=0)

     def transform(self, data):
          normalized_data = (data - self.min_val) / (self.max_val - self.min_val)
          return normalized_data

     def fit_transform(self, data):
          self.fit(data)
          normalized_data = self.transform(data)
          return normalized_data

     def inverse_transform(self, data):
          desnormalized_data = data * (self.max_val - self.min_val) + self.min_val
          return desnormalized_data

     def export(self, path):
          import pickle
          with open(path, 'wb') as f:
               pickle.dump(self, f)
     

class ZScoreScaler:
     def __init__(self):
          self.mean = None
          self.std = None

     def get_mean(self):
          return self.mean
     
     def get_std(self):
          return self.std

     def fit(self, data):
          self.mean = np.mean(data, axis=0)
          self.std = np.std(data, axis=0)

     def transform(self, data):
          normalized_data = (data - self.mean) / (self.std)
          return normalized_data

     def fit_transform(self, data):
          self.fit(data)
          normalized_data = self.transform(data)
          return normalized_data

     def inverse_transform(self, data):
          desnormalized_data = data * (self.std) + self.mean
          return desnormalized_data

     def export(self, path):
          import pickle
          with open(path, 'wb') as f:
               pickle.dump(self, f)

class DataSplitter:
     def __init__(self):
          pass
     
     def split_data(self, x, y, test_size=0.2, random_state=None):
          if random_state is not None:
               np.random.seed(random_state)
          shuffled_indices = np.random.permutation(len(x))
          test_set_size = int(len(x) * test_size)
          test_indices = shuffled_indices[:test_set_size]
          train_indices = shuffled_indices[test_set_size:]
          x_train = x[train_indices]
          x_test = x[test_indices]
          y_train = y[train_indices]
          y_test = y[test_indices]
          return x_train, x_test, y_train, y_test

