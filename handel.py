
import numpy as np

# working with axis


# let's create a 2d array

array = [[3, 8.5, 2],
		 [2, 0.6, 4.2],
		 [1, 2, 0.43]]

output = np.sum(array, axis=0);

print(output) # [6, 11, 1, 6.63] so it print the sum of each row because axis=0

output0 = np.sum(array, axis=1, keepdims=True);

print(output0)



# craete a sofmax activation function class

class softmax_activation_function:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities



softmax_output = softmax_activation_function()

softmax_output.forward([[1, 2, 3]])
print(softmax_output.output)
print(np.sum(softmax_output.output))