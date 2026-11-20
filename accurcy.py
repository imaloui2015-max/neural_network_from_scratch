import numpy as np




output = np.array([[0.6, 0.1, 0.3],
		  [0.3, 0.5, 0.2],
		  [0.1, 0.9, 0]])


class_targets = np.array([0, 1, 1])


predictions = np.argmax(output, axis=1)

if (len(class_targets.shape) == 2):
	class_targets = np.argmax(class_targets, axis=1)

accurcy = np.mean(predictions == class_targets)

print(f"acc is: {accurcy}")