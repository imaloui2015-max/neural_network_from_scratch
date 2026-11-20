
import numpy as np
import nnfs
import matplotlib.pyplot as plt

from nnfs.datasets import spiral_data, vertical_data


nnfs.init()


# data
X, y = vertical_data(samples=100, classes=3)


# initialize out parameters and create a forward pass for the neuron that takes an inputs
class Layer_Dense():
  def __init__(self, n_inputs, n_neurons):
    self.wieghts = np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    self.output = np.dot(inputs, self.wieghts) + self.biases



# Shose an activation function and pass whatever the neuron output it in this case where using the Relu that stands for rectifide linear unit
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# and using softmax for that takes logits and output probabilities after applying expononial and substrict by the max then devide by the total
class activation_softmax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # axis equal to one refers to column
    probabilities  = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    self.output = probabilities


# Calculating the Loss
class Loss:
  def calculate(self, output, y):
    sample_loss = self.forward(output, y)
    average_loss = np.mean(sample_loss)
    return average_loss


# using categoricalcrossentropy for calculating the loss
class CategoricalCrossEntropy(Loss):
  def forward(self, y_pred, y_true):
    samples = len(y_pred)
    y_pred_clips = np.clip(y_pred, 1e-7, 1-1e-7)

    if len(y_true.shape) == 1:
      correct_confident = y_pred_clips[range(samples), y_true]

    elif len(y_true.shape) == 2:
      correct_confident = np.sum(y_pred_clips * y_true, axis=1)

    likelihoods = -np.log(correct_confident)
    return likelihoods

    





# so now we are going to use all the stuff that we ceate above

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = activation_softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


loss_fn = CategoricalCrossEntropy()
loss = loss_fn.calculate(activation2.output, y)

print("The Loss Is: ", loss)

predictions = np.argmax(activation2.output, axis=1)

if len(y.shape) == 2:
  y = np.argmax(y, axis=1)

print(np.mean(predictions==y))



# colored data
#plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
#plt.show()










#layer1 = Layer_Dense(len(X[0]), 5)
#activation1 = Activation_ReLU()

#layer1.forward(X)


#activation1.forward(layer1.output)
#print(activation1.output)




