import numpy as np

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    #np.random.seed(0)
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    # self.biases = np.zeros((1, n_neurons))
    self.biases = 0.01 * np.random.randn(1, n_neurons)
    self.n_inputs = n_inputs
    self.n_neurons = n_neurons

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