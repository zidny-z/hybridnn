import numpy as np

# Define BPNN class
class Backpropagation:
     def __init__(self, input_size, hidden_sizes, activation_functions, output_size, learning_rate, enable_bias=False):
          self.input_size = input_size
          self.hidden_sizes = hidden_sizes
          self.output_size = output_size
          self.learning_rate = learning_rate
          self.weights = []
          self.enable_bias = enable_bias
          self.biases = []
          self.activations = activation_functions
          self.error_track = []

          # initialize weights and biases
          sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
          for i in range(len(sizes) - 1):
               self.weights.append(np.random.randn(sizes[i], sizes[i+1]))
               if self.enable_bias:
                    self.biases.append(np.random.randn(sizes[i+1]))

     def get_error_track(self):
          return self.error_track
     
     def get_bias(self):
          return self.biases

     def get_activations(self):
          activation_names = [type(act).__name__ for act in self.activations]
          return ", ".join(activation_names)

     def forward(self, x):
          a = x
          for i in range(len(self.weights)):
               if self.enable_bias:
                    z = np.dot(a, self.weights[i]) + self.biases[i]
               else:
                    z = np.dot(a, self.weights[i])
               a = self.activations[i].forward(z)
          return a

     def backward(self, x, y):
          a = x
          zs = []
          activations = [a]
          for i in range(len(self.weights)):
               if self.enable_bias:
                    z = np.dot(a, self.weights[i]) + self.biases[i]
               else:
                    z = np.dot(a, self.weights[i])
               zs.append(z)
               a = self.activations[i].forward(z)
               activations.append(a)

          delta = (activations[-1] - y) * self.activations[-1].backward(zs[-1])
          deltas = [delta]
          for i in range(len(self.weights) - 1, 0, -1):
               delta = np.dot(deltas[-1], self.weights[i].T) * self.activations[i-1].backward(zs[i-1])
               deltas.append(delta)
          deltas.reverse()

          gradients = []
          for i in range(len(self.weights)):
               gradient_w = np.dot(activations[i].T, deltas[i])
               if self.enable_bias:
                    gradient_b = np.sum(deltas[i], axis=0)
                    gradients.append((gradient_w, gradient_b))
               else:
                    gradients.append((gradient_w, None))

          return gradients

     def update_weights(self, gradients):
          for i in range(len(self.weights)):
               self.weights[i] -= self.learning_rate * gradients[i][0]
               if self.enable_bias:
                    self.biases[i] -= self.learning_rate * gradients[i][1]
               # self.biases[i] -= self.learning_rate * gradients[i][1]

     def train(self, x, y, epochs):
          for epoch in range(epochs):
               for i in range(len(x)):
                    gradients = self.backward(x[i:i+1], y[i:i+1])
                    self.update_weights(gradients)
               # tracking eror
               y_pred = self.predict(x)
               error = self.rmse(y_pred, y)
               self.error_track.append(error)

     def predict(self, x):
          return self.forward(x)

     def rmse(self, y_pred, y_true):
          return np.sqrt(np.mean(np.square(y_pred - y_true)))
     
     def mape(self, y_pred, y_true):
          return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     
     def mae(self, y_pred, y_true):
          return np.mean(np.abs(y_true - y_pred))

     # export model to pickle
     def export_model(self, filename):
          import pickle
          with open(filename, 'wb') as file:
               pickle.dump(self, file)
     
# Define PSO class
class Particle:
     def __init__(self, input_size, hidden_sizes, activation_functions, output_size, learning_rate):
          self.position = []
          self.velocity = []
          self.best_position = []
          self.best_error = float('inf')
          self.error = float('inf')
          self.bpnn = Backpropagation(input_size, hidden_sizes, activation_functions, output_size, learning_rate)

     def initialize(self):
          for i in range(len(self.bpnn.weights)):
               self.position.append(np.random.randn(*self.bpnn.weights[i].shape))
               self.velocity.append(np.zeros_like(self.bpnn.weights[i]))

     def update_velocity(self, global_best_position, c1, c2, w):
          for i in range(len(self.velocity)):
               r1 = np.random.rand(*self.bpnn.weights[i].shape)
               r2 = np.random.rand(*self.bpnn.weights[i].shape)
               cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
               social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
               self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

     def update_position(self):
          for i in range(len(self.position)):
               self.position[i] += self.velocity[i]
               self.bpnn.weights[i] = self.position[i]

     def evaluate(self, x, y):
          y_pred = self.bpnn.predict(x)
          self.error = self.bpnn.rmse(y_pred, y)
          if self.error < self.best_error:
               self.best_position = self.position
               self.best_error = self.error

class NNPSO:
     def __init__(self, input_size, hidden_sizes, activation_functions, output_size, learning_rate, num_particles, num_iterations, w):
          self.num_particles = num_particles
          self.num_iterations = num_iterations
          self.c1 = learning_rate
          self.c2 = learning_rate
          self.w = w
          self.particles = []
          self.global_best_position = []
          self.global_best_error = float('inf')
          self.bpnn = Backpropagation(input_size, hidden_sizes, activation_functions, output_size, learning_rate)
          self.error_track = []

     def get_error_track(self):
          return self.error_track

     def initialize(self):
          for i in range(self.num_particles):
               particle = Particle(self.bpnn.weights[0].shape[0], [self.bpnn.weights[i].shape[1] for i in range(len(self.bpnn.weights)-1)], self.bpnn.activations, self.bpnn.weights[-1].shape[1], self.bpnn.learning_rate)
               particle.initialize()
               self.particles.append(particle)
          self.global_best_position = self.particles[0].position

     def optimize(self, x, y):
          for i in range(self.num_iterations):
               for particle in self.particles:
                    particle.evaluate(x, y)
                    if particle.error < self.global_best_error:
                         self.global_best_position = particle.position
                         self.global_best_error = particle.error
               for particle in self.particles:
                    particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w)
                    particle.update_position()
               self.error_track.append(self.global_best_error)
          
     def get_weights(self):
          return self.global_best_position

     def predict(self, x):
          for i in range(len(self.global_best_position)):
               self.bpnn.weights[i] = self.global_best_position[i]
          return self.bpnn.predict(x)

     def rmse(self, y_pred, y_true):
          return np.sqrt(np.mean(np.square(y_pred - y_true)))
     
     def mape(self, y_pred, y_true):
          return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     
     def mae(self, y_pred, y_true):
          return np.mean(np.abs(y_true - y_pred))

     def export_model(self, filename):
          import pickle
          with open(filename, 'wb') as file:
               pickle.dump(self, file)