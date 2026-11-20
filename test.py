import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2, 5, -1, 2],
    [-1.5, 2.7, 3.3, -0.8]]


inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []


'''
for i in inputs:
	if i > 0:
		output.append(i)
	else:
		output.append(0)
'''
for i in inputs:
	output.append(max(0, i))

# print(output)


# binary cross entropy


softmax_output = [0.7, 0.1, 0.2]

target_value = [1, 0, 0]
import math

loss = -(math.log(softmax_output[0])*target_value[0] +
		math.log(softmax_output[1])*target_value[1] +
		math.log(softmax_output[2])*target_value[2])

print(loss)
