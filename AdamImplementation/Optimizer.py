import numpy as np
from LayerDense import *

# SGD optimizer
class Optimizer_SGD:
  # Initialize optimizer - set settings,
  # learning rate of 1. is default for this optimizer
  def __init__(self, learning_rate=1.0):
    self.learning_rate = learning_rate

  def update_params(self, layer:Layer_Dense): 
    layer.weights += - self.learning_rate * layer.dweights
    layer.biases += - self.learning_rate * layer.dbiases


class Optimizer:
  def __init__(self, learning_rate:float=1.0, decay:float=0.):
    """
        Params:
    """
    self.learning_rate = learning_rate
    self.decay = decay # learning rate decay rate
    self.current_learning_rate = learning_rate
    self.step = 0 # stores the current epoch
    
  def pre_update_parameters(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.step))

  def post_update_params(self):
      self.step += 1


class Optimizer_SGD_Decay(Optimizer):
  """
  SGD optimizer with learning rate decay.
  """
    
  def update_params(self, layer:Layer_Dense):
    layer.weights += - self.current_learning_rate * layer.dweights
    layer.biases += - self.current_learning_rate * layer.dbiases


class Optimizer_AdaGrad(Optimizer):
  """
  AdaGrad optimizer with adaptative parameter updates by keeping a history of previous updates. 
  """  
  def __init__(self, learning_rate:float=1.0, decay:float=0., epsilon:float=1e-7):
    super().__init__(learning_rate,decay)
    self.epsilon = epsilon # prevents dividing by zero

  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)

    # update caches with the current gradients
    layer.weight_cache += layer.dweights**2 
    layer.biases_cache += layer.dbiases**2

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)


class Optimizer_RMSProp(Optimizer):
  """
  RMSProp (Root Mean Square Propagation) optimizer adds a mechanism similar to momentum and also adds an adaptive learning rate 
  (per-parameter), so the learning rate changes are smoother. 
  It uses a moving average of the cache, rather than the squared gradient.
  """
  def __init__(self, learning_rate:float=0.001, decay:float=0., rho:float=0.9, epsilon:float=1e-7):
    super().__init__(learning_rate,decay)
    self.rho = rho # cache memory decay rate 
    self.epsilon = epsilon # prevents dividing by zero

  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)

    # update caches with the current gradients, in a average sense
    layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2 
    layer.biases_cache = self.rho * layer.biases_cache + (1 - self.rho) * layer.dbiases**2

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)

class Optimizer_Adam(Optimizer):
  """
  Adam (Adaptive Momentum) optimizer similar to the RMSProp update, amplified with Momentum.
  Momentum: Uses the previous update's direction to influence the next update's direction, 
            minimizing the chances of getting stuck in a local minima.
  It also adds a bias correction mechanism applied to the cache and momentum.
  """
  def __init__(self, learning_rate:float=0.001, decay:float=0., beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-7):
    super().__init__(learning_rate,decay)
    self.beta1 = beta1 # momentum decay rate
    self.beta2 = beta2 # cache memory decay rate
    self.epsilon = epsilon # prevents dividing by zero
  
  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache (nor momentum) array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)
      layer.weight_momentum = np.zeros_like(layer.weights)
      layer.biases_momentum = np.zeros_like(layer.biases)

    # update momentum with the current gradients
    layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.dweights 
    layer.biases_momentum = self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dbiases

    # update caches with the current (squared) gradients
    layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights**2 
    layer.biases_cache = self.beta2 * layer.biases_cache + (1 - self.beta2) * layer.dbiases**2

    # correct momentum
    correct_weight_momentum = layer.weight_momentum / (1 - self.beta1**(self.step+1))
    correct_biases_momentum = layer.biases_momentum / (1 - self.beta1**(self.step+1))

    # correct cache
    correct_weight_cache = layer.weight_cache / (1 - self.beta2**(self.step+1))
    correct_biases_cache = layer.biases_cache / (1 - self.beta2**(self.step+1))

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * correct_weight_momentum / (np.sqrt(correct_weight_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * correct_biases_momentum / (np.sqrt(correct_biases_cache) + self.epsilon)

class Optimizer_SGD_Momentum(Optimizer):
  """
  SGD optimizer with momentum for improved convergence.
  Momentum helps accelerate gradients vectors in the right directions,
  thus leading to faster converging.
  """
  def __init__(self, learning_rate:float=1.0, decay:float=0., momentum:float=0.9):
    super().__init__(learning_rate, decay)
    self.momentum = momentum # momentum coefficient

  def update_params(self, layer:Layer_Dense):
    # if layer does not have a momentum array, create it initialized with zeros
    if not hasattr(layer, 'weight_momentum'):
      layer.weight_momentum = np.zeros_like(layer.weights)
      layer.biases_momentum = np.zeros_like(layer.biases)

    # Clip gradients to prevent explosion with high learning rates
    max_grad_norm = 10.0  # Maximum gradient norm
    weight_grad_norm = np.linalg.norm(layer.dweights)
    bias_grad_norm = np.linalg.norm(layer.dbiases)
    
    # Clip weight gradients
    if weight_grad_norm > max_grad_norm:
      layer.dweights = layer.dweights * (max_grad_norm / weight_grad_norm)
    
    # Clip bias gradients  
    if bias_grad_norm > max_grad_norm:
      layer.dbiases = layer.dbiases * (max_grad_norm / bias_grad_norm)

    # Update momentum with current gradients
    layer.weight_momentum = self.momentum * layer.weight_momentum - self.current_learning_rate * layer.dweights
    layer.biases_momentum = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.dbiases

    # Clip momentum to prevent explosion
    max_momentum_norm = 50.0  # Maximum momentum norm
    weight_momentum_norm = np.linalg.norm(layer.weight_momentum)
    bias_momentum_norm = np.linalg.norm(layer.biases_momentum)
    
    if weight_momentum_norm > max_momentum_norm:
      layer.weight_momentum = layer.weight_momentum * (max_momentum_norm / weight_momentum_norm)
      
    if bias_momentum_norm > max_momentum_norm:
      layer.biases_momentum = layer.biases_momentum * (max_momentum_norm / bias_momentum_norm)

    # Update params with momentum
    layer.weights += layer.weight_momentum
    layer.biases += layer.biases_momentum

    # Check for NaN or infinite values
    if np.any(np.isnan(layer.weights)) or np.any(np.isinf(layer.weights)):
      print(f"Warning: NaN or Inf detected in weights. Resetting momentum.")
      layer.weight_momentum = np.zeros_like(layer.weights)
      
    if np.any(np.isnan(layer.biases)) or np.any(np.isinf(layer.biases)):
      print(f"Warning: NaN or Inf detected in biases. Resetting momentum.")
      layer.biases_momentum = np.zeros_like(layer.biases)
