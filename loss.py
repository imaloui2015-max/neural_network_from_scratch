

import numpy as np
import math


softmax_output = [0.7, 0.1, 0.2]

target_value = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_value[0] +
		 math.log(softmax_output[1]) * target_value[1] +
		 math.log(softmax_output[2]) * target_value[2])


actuall = -math.log(softmax_output[0])

print(loss == actuall)

print(loss)


# see an exemple for a batches of softmax output labled by some element like 0 or 1 (dog or cat)

output = [[0.6, 0.1, 0.3],
		  [0.3, 0.5, 0.2],
		  [0.1, 0.9, 0]]


target_class = [0, 1, 1]

for target_idx, output_data in zip(target_class, output):
	print(output_data[target_idx])

# using numpy

output_np = np.array([[0.6, 0.1, 0.3],
		  [0.3, 0.5, 0.2],
		  [0.1, 0.9, 0]])

print(output_np[[0, 1, 2], target_class])


print(output_np[range(len(output_np)), target_class])

# Now w gonna applie the log

print(-np.log(output_np[range(len(output_np)), target_class]))

# checking if the log output is a hot encoded or is list of lis (2 dimension array)

softmax_output = np.array([[0.6, 0.1, 0.3],
		  		  [0.3, 0.5, 0.2],
		  		  [0.1, 0.9, 0]])



class_targets = np.array([[1, 0, 0],
				 [0, 1, 0],
				 [0, 1, 0]])



if len(class_targets.shape) == 1:
	correct_confident = softmax_output[range(len(softmax_output)), target_class]

elif len(class_targets.shape) == 2:
	correct_confident = np.sum(

					softmax_output * class_targets,
					axis=1
	)


# loss
cross_entrop_loss = -np.log(correct_confident)
print(cross_entrop_loss)






