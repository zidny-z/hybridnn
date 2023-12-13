#  make sure to install the following packages
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from hybridnn import Backpropagation, NNPSO, Sigmoid, Tanh, MinMaxScaler, DataSplitter 

# Load data
data = pd.read_csv('dataset.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)


# normalization with min max scaler
X_normalization = MinMaxScaler()
X = X_normalization.fit_transform(x)

Y_normalization = MinMaxScaler()
Y = Y_normalization.fit_transform(y)

# Split data
splitter = DataSplitter()
x_train, x_test, y_train, y_test = splitter.split_data(X, Y, test_size=0.20, random_state=12)

# model parameters
input_size = x_train.shape[1]
hidden_sizes = [7]
activation_functions = [Tanh(), Sigmoid()]
output_size = y_train.shape[1]
learning_rate = .7
num_particles = 60
num_iterations = 30
w = .6

bpnn = Backpropagation(input_size, hidden_sizes,activation_functions, output_size, learning_rate)
start_time = time.time()
bpnn.train(x_train, y_train, num_iterations)
bpnn_computing_time = time.time() - start_time
bpnn_error_track = bpnn.get_error_track()

# # Predict and calculate error
bpnn_y_pred = bpnn.predict(x_test)
bpnn_error = bpnn.rmse(bpnn_y_pred, y_test)

nnpso = NNPSO(input_size, hidden_sizes, activation_functions, output_size, learning_rate, num_particles, num_iterations, w)
nnpso.initialize()
start_time = time.time()
nnpso.optimize(x_train, y_train)
nnpso_computing_time = time.time() - start_time
nnpso_error_track = nnpso.get_error_track()

# Predict and calculate error
nnpso_y_pred = nnpso.predict(x_test)
nnpso_error = nnpso.rmse(nnpso_y_pred, y_test)

improve_error = bpnn_error - nnpso_error
improve_error_percent = (improve_error/bpnn_error)*100

improve_time = bpnn_computing_time - nnpso_computing_time
improve_time_percent = (improve_time/bpnn_computing_time)*100

print('eror bpnn :', bpnn_error)
print('eror nnpso :', nnpso_error)
print('improve error percent :', improve_error_percent)
print('improve computing time percent :', improve_time_percent)

# plot error track bpnn and nnpso
plt.plot(bpnn_error_track, label='BPNN')
plt.plot(nnpso_error_track, label='NNPSO')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()

# plot y test and y pred
plt.plot(y_test, label='Actual')
plt.plot(bpnn_y_pred, label='BPNN')
plt.plot(nnpso_y_pred, label='NNPSO')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.show()



