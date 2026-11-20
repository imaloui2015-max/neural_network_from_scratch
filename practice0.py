import numpy as np
import matplotlib.pyplot as plt

# Create Softmax activation fonction
class softmax_activation_function:
	def forward(self, inputs):
		ans = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = ans / np.sum(ans, axis=1, keepdims=True)
		self.output = probabilities

inputs = [[5, 0.3, 1],
		  [0.2, 6, 1],
		  [3, 2, -0.3]]
softmax_activation_fn = softmax_activation_function()
softmax_activation_fn.forward(inputs)
print(softmax_activation_fn.output)


