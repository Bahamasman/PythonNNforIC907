import numpy as np
import matplotlib.pyplot as plt

## File to contain all classes for the neural network
class HiddenLayer:
  def createHiddenLayer(self,size,activationFunc):
    self.size = size

    if activationFunc == 'ReLU':
      self.activationf  = Activation_ReLU()
    
    elif activationFunc == 'LeakReLU':
      #implement me
      raise NameError('Activation_func function' + activationFunc + 'not supported yet!')
    
    elif activationFunc == 'sigmoid':
      self.activationf = Activation_Sigmoid()
    
    elif activationFunc == 'tanh':
      #implement me
      raise NameError('Activation_func function' + activationFunc + 'not supported yet!')
   
    else:
      raise NameError('Activation_func function' + activationFunc + 'not supported yet!')
  
class myNeuralNetwork():
  #This class is based of our original class when using pyTorch
  def __init__(self):
    self.inputSize = 0
    self.hiddenLayers = []
    self.outputSize = 0
    self.ModulesList = []
  
  def addInputSize(self,size):
      self.inputSize = size

  def addOutputSize(self,size):
      self.output_size = size

  def addHiddenLayer(self,size,activationFunc):
    newHiddenLayer = HiddenLayer()
    newHiddenLayer.createHiddenLayer(size,activationFunc)
    self.hiddenLayers.append(newHiddenLayer)

  def definieLossFunction(self):
    #Update me, add different loss function similar to adding activation function
    self.lossf = Loss_MeanSquaredError()

  def defineOptimizer(self,learningRate):
    #Update me, add different optimizers similar to adding activation function
    self.optim = Optimizer_SGD(learningRate)

  def build(self,inputSize,outputSize,numHiddenLayers,hiddenSize,activationFunc,rate=1.0):
    self.addInputSize(inputSize)
    self.addOutputSize(outputSize)

    if type(hiddenSize) == list:
      if len(hiddenSize) != numHiddenLayers:
        raise NameError(f'Number of hidden layers is {numHiddenLayers}, provided size for {len(hiddenSize)} layers. Must provide size for each layer!')
      for i in range(numHiddenLayers):
        self.addHiddenLayer(hiddenSize[i],activationFunc)

    else:
      scale = 1.0
      for i in range(numHiddenLayers):
          self.addHiddenLayer(int(np.ceil(hiddenSize * scale)),activationFunc)
          scale = scale * rate

  def intialize(self):
    inputSize = self.inputSize

    for hiddenLayer in self.hiddenLayers:
      self.ModulesList.append(Layer_Dense(inputSize,hiddenLayer.size))
      inputSize = hiddenLayer.size

    self.ModulesList.append(Layer_Dense(inputSize,self.output_size))


  def forward(self,input):
    out = input
    for index, hiddenLayer in enumerate(self.hiddenLayers):
        
        self.ModulesList[index].forward(out)
        out = self.ModulesList[index].output
        
        hiddenLayer.activationf.forward(out)
        out = hiddenLayer.activationf.output
    
    ff = self.ModulesList[-1]
    ff.forward(out)
    self.output = ff.output
    
  def computeLoss(self,yreal):

    return self.lossf.calculate(self.output,yreal)
    

  def backward(self,yreal):
    
    self.lossf.backward(self.output,yreal)
    dinputs = self.lossf.dinputs

    ff = self.ModulesList[-1]
    ff.backward(dinputs)
    dinputs = ff.dinputs

    for index, hiddenLayer in reversed(list(enumerate(self.hiddenLayers))):
        
        hiddenLayer.activationf.backward(dinputs)
        dinputs = hiddenLayer.activationf.dinputs

        self.ModulesList[index].backward(dinputs)
        dinputs = self.ModulesList[index].dinputs
    
    self.dinputs = dinputs

  def updateParam(self):
    for layer in self.ModulesList :
      self.optim.update_params(layer)



### Utility Classes ####

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

