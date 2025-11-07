import numpy as np

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



class Activation_Sigmoid:

  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs #is this used???

    # Calculate output values from inputs
    self.output = 1/(1+np.exp(-self.inputs))
  
  # Backward pass
  def backward(self, dvalues):
    self.dinputs = dvalues.copy()

    self.dinputs = self.dinputs * self.output * (1 - self.output)