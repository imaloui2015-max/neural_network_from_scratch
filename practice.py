import numpy as np
import matplotlib.pyplot as plt
import nnfs

from nnfs.datasets import spiral_data

X, y = spiral_data(100, 3)

nnfs.init()

'''


class Layer_Dense():
	def __init__(self, n_inputs, n_neurons):
		self.weight = np.random.randn(n_inputs, n_neurons)
		self.bias = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weight) + self.bias


class Activation_ReLU():
	def forward(self, inputs):
		self.output = np.max(0, inputs);


layer1 = Layer_Dense(len(X[0]), 5)
layer1.forward(X)
print(layer1.output)




'''

np.random.seed(42)

n_inputs = 5
n_neurons = 4
print(np.random.randn(n_inputs, n_neurons))


weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))

# print(weights)
# print(biases)

# adding bias

weights  += biases
# print(weights)

output = [[0, 1, 2, 2],
		  [3, 4, 4, 4]]

biases = [[1, 1, 1, 1]]

inputs = [[1, 1, 1, 1]]

# res = np.dot(inputs, output) + biases => astype attribue error 'list' object has no attribue
# print(res)

class Layer_Dense():
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases



#dense1 = Layer_Dense(2, 3)
#dense1.forward(X)


#print(X)






