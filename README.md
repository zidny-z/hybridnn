## HybridNN
HybridNN is a Package for Backpropagation and Hybrid Neural Netwok with Particle Swarm Optimization

### Description
HybridNN is a Package for Backpropagation and Hybrid Neural Netwok with Particle Swarm Optimization with these feature:
- Data normalization with min-max scaler and z-score scaler
- Data splitter
- BPNN and NNPSO algorthm with export model and get error track

### Install
To install HybridNN, you can use pip:
```
pip install hybridnn
```

### Example
Normalization data and split data
```
from hybridnn import MinMaxScaler, DataSplitter 

X_normalization = MinMaxScaler()
X = X_normalization.fit_transform(x)
Y_normalization = MinMaxScaler()
Y = Y_normalization.fit_transform(y)

# Split data to 80% data train and 20% data test
splitter = DataSplitter()
x_train, x_test, y_train, y_test = splitter.split_data(X, Y, test_size=0.20, random_state=12)
```
Train backpropagation model
```
from hybridnn import Backpropagation, Tanh, Sigmoid 
bpnn = Backpropagation(4, [2],[Tanh(),Sigmoid()], 1, .1)
bpnn.train(x_train, y_train, num_iterations)

# Predict and calculate error
bpnn_y_pred = bpnn.predict(x_test)
bpnn_error = bpnn.rmse(bpnn_y_pred, y_test)
print('bpnn error :', bpnn_error)
```
Train NNPSO (hybrid neural network with PSO)
```
from hybridnn import NNPSO, Tanh, Sigmoid
nnpso = NNPSO(4, [2], [Tanh(), Sigmoid()], 1, .1, 30, 25, .5)
nnpso.initialize()
nnpso.optimize(x_train, y_train)

# Predict and calculate error
nnpso_y_pred = nnpso.predict(x_test)
print('nnpso error :', nnpso_y_pred)
```
Export Model
```
bpnn.export_model('bpnn.pkl')
```
Get Error Track
```
bpnn.get_error_track()
```
#### More Example
You can find more examples in the "example" folder of this HybridNN package.

### Licence

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

### Additional
This package was created for my thesis research, some functions may not be effective for industrial needs