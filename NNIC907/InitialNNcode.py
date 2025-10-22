import numpy as np
import matplotlib.pyplot as plt

class Activation_ReLU:
  # Forward pass
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs
    # Calculate output values from inputs
    self.output = np.maximum(0, inputs)

  # Backward pass
  def backward(self, dvalues):
    # Gradient of ReLU is 1 for positive inputs, 0 for negative inputs
    self.dinputs = dvalues.copy() # Copy to avoid modifying the original array
    # Zero gradient where output was less than or equal to 0
    self.dinputs[self.inputs <=0]=0


class Layer_Dense:

  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    # self.biases = np.zeros((1, n_neurons))
    self.biases = 0.01 * np.random.randn(1, n_neurons)

  # Forward pass
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs


    # Calculate output values from inputs, weights and biases
    self.output = inputs @ self.weights + self.biases

  # Backward pass
  def backward(self, dvalues):
    self.dinputs = dvalues @ self.weights.T
    self.dweights = self.inputs.T @ dvalues
    self.dbiases = np.sum(dvalues ,axis=0,keepdims=True)

# Father class (for any type of loss)
class Loss:

  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):

    # Calculate sample losses
    sample_losses = self.forward(output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Return loss
    return data_loss

# MSE loss class
class Loss_MeanSquaredError(Loss):
  def forward(self, y_pred, y_true):

    # Calculate loss
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

    return sample_losses

  def backward(self,dvalues,y_true): #note that dvalues = y_pred
    nsamples = len(dvalues)
    noutputs = len(dvalues[0])

    # Recall
    # Loss = 1/nsamples * sum(L_i)
    # L_i = 1/noutputs * sum((y_true_i - y_pred_i)**2)
    self.dinputs =  -2 * (y_true - dvalues) / noutputs# Gradient of outputs
    self.dinputs = self.dinputs / nsamples  # normalized by the samples

# SGD optimizer
class Optimizer_SGD:
  # Initialize optimizer - set settings,
  # learning rate of 1. is default for this optimizer
  def __init__(self, learning_rate=1.0):
    self.learning_rate = learning_rate
    # Update parameters
  def update_params(self, layer): #layer is a Layer Dense object
    layer.weights += - self.learning_rate * layer.dweights
    layer.biases +=  - self.learning_rate * layer.dbiases

# Real a and b for the line
xpts = np.linspace(0,1,100)
npts = len(xpts)

# Generating data
yreal = np.sin(2*np.pi*xpts)

# Learning rate and number of epochs
lr = 0.1
epochs = 10001
lossvec = np.zeros(epochs)

# Initialize layer, loss and optimizer
# Propose your own number of hidden layers and number of neurons!
# Create here dense layers, activation layer, loss and optimizer. Don't forget you need different activation objects for different layers
layer1 = Layer_Dense(1,10)
activation1 = Activation_ReLU()

layer2 = Layer_Dense(10,5)
activation2 = Activation_ReLU()

layer3 = Layer_Dense(5,1)

lossf = Loss_MeanSquaredError()
optim = Optimizer_SGD(learning_rate=lr)

# Convert data to our data structure
# Implement using reshape
xptsbatch = xpts.reshape(npts,1)
yrealbatch = yreal.reshape(npts,1)

# Training loop
for epoch in range(epochs):
  # Forward pass
  layer1.forward(xptsbatch)
  activation1.forward(layer1.output)

  layer2.forward(activation1.output)
  activation2.forward(layer2.output)

  layer3.forward(activation2.output)

  # Compute loss (mean squared error)
  lossk = lossf.calculate(layer3.output,yrealbatch)

  # Compute the derivatives
  lossf.backward(layer3.output, yrealbatch)
  layer3.backward(lossf.dinputs)
  activation2.backward(layer3.dinputs)
  layer2.backward(activation2.dinputs)

  activation1.backward(layer2.dinputs)
  layer1.backward(activation1.dinputs)

  # Gradient descent step
  optim.update_params(layer3)
  optim.update_params(layer2)
  optim.update_params(layer1)

  # Store loss and parameters for plotting
  lossvec[epoch] = lossk

  # Print every 500 epochs
  if epoch % 500 == 0:
    print(f"epoch {epoch}: loss = {lossk:.10f}")

# Create a 1x2 subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Left: Data and model fit
axs[0].plot(xpts, yreal, 'o', label='true data')
axs[0].plot(xpts, layer3.output, '-', label='NN prediction')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Sine fit with neural net')
axs[0].legend()

# Right: Loss over epochs
axs[1].plot(lossvec, label='Loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_title('Loss over epochs')
axs[1].legend()

plt.tight_layout()
plt.show()
