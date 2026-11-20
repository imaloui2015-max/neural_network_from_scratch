import math
import numpy as np

layer_output = [[4.8, 1.21, 2.385],
				[8.9, -1.81, 0.2],
				[1.41, 1.051, 0.026]]

exp_values = np.exp(layer_output)

print(exp_values)
print()
normalization_value = exp_values / np.sum(layer_output, axis=1, keepdims=True)

print(normalization_value)


















#print(np.sum(layer_output, axis=1, keepdims=True))

# E = 2.71818281846

#E = math.e

#normalization_value = exp_values / np.sum(exp_values)

#print(normalization_value)
#print(sum(normalization_value))

class_targets = [[1, 0, 0],
				 [0, 1, 0],
				 [0, 1, 0]]

#print(class_targets.shape())

class_targets = np.array([[1, 0, 0],
				 [0, 1, 0],
				 [0, 1, 0]])

print(class_targets)