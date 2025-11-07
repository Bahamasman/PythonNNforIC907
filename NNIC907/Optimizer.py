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



class Optimizer_SGD:
  """
  SGD optimizer with learning rate decay.
  """

  def __init__(self, learning_rate:float=1.0, decay:float=0.):
    """
        Params:
    """
    self.learning_rate = learning_rate
    self.decay = decay
    self.current_learning_rate = learning_rate
    self.step = 0 # to store the current epoch

  # apply decay to learning_rate before each time the parameters are updated
  def pre_update_parameters(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.step))
    
  def update_params(self, layer:Layer_Dense):
    layer.weights += - self.current_learning_rate * layer.dweights
    layer.biases += - self.current_learning_rate * layer.dbiases

  def post_update_params(self):
    self.step += 1



class Optimizer_AdaGrad:
  """
  AdaGrad optimizer with adaptative parameter updates by keeping a history of previous updates. 
  """

  def __init__(self, learning_rate:float=1.0, decay:float=0., epsilon:float=1e-7):
    """
        Params:
    """
    self.learning_rate = learning_rate
    self.decay = decay
    self.current_learning_rate = learning_rate
    self.epsilon = epsilon # prevents dividing by zero
    self.step = 0 # stores the current epoch

  # apply decay to learning_rate before each time the parameters are updated
  def pre_update_parameters(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.step))
    
  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)

    # update caches with the current gradients
    layer.weights_cache += layer.dweights**2 
    layer.biases_cache += layer.dbiases**2

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weights_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)

  def post_update_params(self):
    self.step += 1



class Optimizer_RMSProp:
  """
  RMSProp (Root Mean Square Propagation) optimizer adds a mechanism similar to momentum and also adds an adaptive learning rate 
  (per-parameter), so the learning rate changes are smoother. 
  It uses a moving average of the cache, rather than the squared gradient.
  """

  def __init__(self, learning_rate:float=0.001, decay:float=0., rho:float=0.9, epsilon:float=1e-7):
    """
        Params:
    """
    self.learning_rate = learning_rate
    self.decay = decay # learning rate decay rate
    self.current_learning_rate = learning_rate
    self.rho = rho # cache memory decay rate 
    self.epsilon = epsilon # prevents dividing by zero
    self.step = 0 # stores the current epoch

  # apply decay to learning_rate before each time the parameters are updated
  def pre_update_parameters(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.step))
    
  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)

    # update caches with the current gradients, in a average sense
    layer.weights_cache += self.rho * layer.weights_cache + (1 - self.rho) * layer.dweights**2 
    layer.biases_cache += self.rho * layer.biases_cache + (1 - self.rho) * layer.dbiases**2

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * layer.dweights / (np.sqrt(layer.weights_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache) + self.epsilon)

  def post_update_params(self):
    self.step += 1



class Optimizer_Adam:
  """
  Adam (Adaptive Momentum) optimizer similar to the RMSProp update, amplified with Momentum.
  Momentum: Uses the previous update's direction to influence the next update's direction, 
            minimizing the chances of getting stuck in a local minima.
  It also adds a bias correction mechanism applied to the cache and momentum.
  """

  def __init__(self, learning_rate:float=0.001, decay:float=0., beta1:float=0.9, beta2:float=0.999, epsilon:float=1e-7):
    """
        Params:
    """
    self.learning_rate = learning_rate
    self.decay = decay # learning rate decay rate
    self.current_learning_rate = learning_rate
    self.beta1 = beta1 # momentum decay rate
    self.beta2 = beta2 # cache memory decay rate
    self.epsilon = epsilon # prevents dividing by zero
    self.step = 0 # stores the current epoch

  # apply decay to learning_rate before each time the parameters are updated
  def pre_update_parameters(self):
    if self.decay:
      self.current_learning_rate = self.learning_rate * (1./(1. + self.decay*self.step))
    
  def update_params(self, layer:Layer_Dense):
    # if layer does not have a cache (nor momentum) array, create it initialized with zeros
    if not hasattr(layer, 'weight_cache'):
      layer.weight_cache = np.zeros_like(layer.weights)
      layer.biases_cache = np.zeros_like(layer.biases)
      layer.weight_momentum = np.zeros_like(layer.weights)
      layer.biases_momentum = np.zeros_like(layer.biases)

    # update momentum with the current gradients
    layer.weights_momentum += self.beta1 * layer.weights_momentum + (1 - self.beta1) * layer.dweights 
    layer.biases_momentum += self.beta1 * layer.biases_momentum + (1 - self.beta1) * layer.dweights

    # update caches with the current (squared) gradients
    layer.weights_cache += self.beta2 * layer.weights_cache + (1 - self.beta2) * layer.dweights**2 
    layer.biases_cache += self.beta2 * layer.biases_cache + (1 - self.beta2) * layer.dbiases**2

    # correct momentum
    correct_weights_momentum = layer.weights_momentum / (1 - self.beta1**(self.step+1))
    correct_biases_momentum = layer.biases_momentum / (1 - self.beta1**(self.step+1))

    # correct cache
    correct_weights_cache = layer.weights_cache / (1 - self.beta2**(self.step+1))
    correct_biases_cache = layer.biases_cache / (1 - self.beta2**(self.step+1))

    # update params with normalized gradient
    layer.weights += - self.current_learning_rate * correct_weights_momentum / (np.sqrt(correct_weights_cache) + self.epsilon)
    layer.biases += - self.current_learning_rate * correct_biases_momentum / (np.sqrt(correct_biases_cache) + self.epsilon)

  def post_update_params(self):
    self.step += 1