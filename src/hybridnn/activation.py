import numpy as np

class Sigmoid:
     def __init__(self):
          pass
      
     def forward(self, x):
          return 1 / (1 + np.exp(-x))
      
     def backward(self, x):
          return self.forward(x) * (1 - self.forward(x))

class ReLU:
     def __init__(self):
          pass
      
     def forward(self, x):
          return np.maximum(0, x)
      
     def backward(self, x):
          return np.where(x > 0, 1, 0)

class Tanh:
     def __init__(self):
          pass
      
     def forward(self, x):
          return np.tanh(x)
      
     def backward(self, x):
          return 1 - np.square(self.forward(x))

class Softmax:
     def __init__(self):
          pass
      
     def forward(self, x):
          return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
      
     def backward(self, x):
          return self.forward(x) * (1 - self.forward(x))

class Linear:
     def __init__(self):
          pass
      
     def forward(self, x):
          return x
      
     def backward(self, x):
          return np.ones_like(x)

class Swish:
     def __init__(self):
          pass
      
     def forward(self, x):
          return x * Sigmoid().forward(x)
      
     def backward(self, x):
          return Sigmoid().forward(x) + x * Sigmoid().backward(x)

class Mish:
     def __init__(self):
          pass
      
     def forward(self, x):
          return x * np.tanh(np.log(1 + np.exp(x)))
      
     def backward(self, x):
          return np.tanh(np.log(1 + np.exp(x))) + x * (1 - np.square(np.tanh(np.log(1 + np.exp(x))))) * (1 / (1 + np.exp(-x)))

class Elu:
     def __init__(self):
          pass
      
     def forward(self, x):
          return np.where(x > 0, x, np.exp(x) - 1)
      
     def backward(self, x):
          return np.where(x > 0, 1, np.exp(x))

class Selu:
     def __init__(self):
          pass
      
     def forward(self, x):
          alpha = 1.6732632423543772848170429916717
          scale = 1.0507009873554804934193349852946
          return scale * np.where(x > 0, x, alpha * np.exp(x) - alpha)
      
     def backward(self, x):
          alpha = 1.6732632423543772848170429916717
          scale = 1.0507009873554804934193349852946
          return scale * np.where(x > 0, 1, alpha * np.exp(x))

class Gelu:
     def __init__(self):
          pass
      
     def forward(self, x):
          return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
      
     def backward(self, x):
          return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))) + 0.5 * x * (1 - np.square(np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))) * (np.sqrt(2 / np.pi) * (1 + 0.134145 * np.power(x, 2)))

