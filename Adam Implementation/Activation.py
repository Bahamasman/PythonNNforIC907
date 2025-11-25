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

    self.dinputs *= self.output * (1 - self.output)
  
class Activation_TanhH:
  
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs
    # Calculate output values from inputs
    self.output = np.tanh(self.inputs)

  # Backward pass
  def backward(self, dvalues):
    # Copy to avoid modifying the original array
    self.dinputs = dvalues.copy()
    # Derivative of tanh is 1 - tanh(x)^2
    self.dinputs *= (1 - self.output ** 2)

class Activation_LeakyReLU:
  
  def __init__(self, alpha=0.01):
    """
    Initialize Leaky ReLU activation function
    
    Parameters:
    alpha: float, slope for negative values (default 0.01)
           Common values: 0.01, 0.1, 0.2
    """
    self.alpha = alpha
  
  def forward(self, inputs):
    # Save inputs for backpropagation
    self.inputs = inputs
    # Calculate output values: max(alpha * x, x)
    # For positive x: output = x
    # For negative x: output = alpha * x
    self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

  # Backward pass
  def backward(self, dvalues):
    # Copy to avoid modifying the original array
    self.dinputs = dvalues.copy()
    
    # Gradient of Leaky ReLU:
    # For positive inputs: gradient = 1
    # For negative inputs: gradient = alpha
    self.dinputs[self.inputs <= 0] *= self.alpha