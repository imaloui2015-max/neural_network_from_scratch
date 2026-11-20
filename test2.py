import numpy as np


def mean(input):
	summ = 0
	for i in range(len(input)):
		summ += input[i]
	return summ / len(input)


softmax_output = np.array([[0.8, 0.1, 0.1],
				   [0.3, 0.5, 0.2],
				   [0.2, 0.7, 0.1]])

class_targets = [0, 1, 1]

summ = 0

print(softmax_output[[0, 1, 2], class_targets])
print(-np.log(softmax_output[[0, 1, 2], class_targets]))
print(np.mean(-np.log(softmax_output[[0, 1, 2], class_targets])))

output = -np.log(softmax_output[[0, 1, 2], class_targets])


for i in range(len(output)):
	summ += output[i]

print(summ / len(output))

print()
print()
print(mean(-np.log(softmax_output[[0, 1, 2], class_targets])))







