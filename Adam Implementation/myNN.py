import numpy as np
import matplotlib.pyplot as plt
from Activation import *
from LayerDense import *
from Loss import *
from Optimizer import *

## File to contain all classes for the neural network
class HiddenLayer:
  def createHiddenLayer(self,size,activationFunc):
    self.size = size

    if activationFunc == 'ReLU':
      self.activationf  = Activation_ReLU()
    
    elif activationFunc == 'LeakyReLU' or activationFunc == 'LeakReLU':
      self.activationf = Activation_LeakyReLU()  # Uses default alpha=0.01
    
    elif activationFunc == 'sigmoid':
      self.activationf = Activation_Sigmoid()
    
    elif activationFunc == 'tanh':
      self.activationf = Activation_TanhH()
   
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
      self.outputSize = size

  def addHiddenLayer(self,size,activationFunc):
    newHiddenLayer = HiddenLayer()
    newHiddenLayer.createHiddenLayer(size,activationFunc)
    self.hiddenLayers.append(newHiddenLayer)

  def defineLossFunction(self):
    #Update me, add different loss function similar to adding activation function
    self.lossf = Loss_MeanSquaredError()

  def defineOptimizer(self,optim,learningRate):
    #Update me, add different optimizers similar to adding activation function

    if optim == 'SGD':
      self.optim = Optimizer_SGD(learningRate)
    elif optim == 'SGD_Decay':
      self.optim = Optimizer_SGD_Decay(learningRate,decay=1e-5)
    elif optim == 'SGD_Momentum':
      self.optim = Optimizer_SGD_Momentum(learningRate,decay=1e-5,momentum=0.9)
    elif optim == 'AdaGrad':
      self.optim = Optimizer_AdaGrad(learningRate,decay=1e-5)
    elif optim == 'RMSProp':
      self.optim = Optimizer_RMSProp(learningRate,decay=1e-5)
    elif optim == 'Adam':
      self.optim = Optimizer_Adam(learningRate,decay=5e-7)
    else:
      raise NameError('Optimizer ' + optim + ' not supported!')

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

  def initialize(self):
    inputSize = self.inputSize

    for hiddenLayer in self.hiddenLayers:
      self.ModulesList.append(Layer_Dense(inputSize,hiddenLayer.size))
      inputSize = hiddenLayer.size

    self.ModulesList.append(Layer_Dense(inputSize,self.outputSize))


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
    return self.output
    
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
    # Check if optimizer has pre_update_parameters method (not all optimizers do)
    if hasattr(self.optim, 'pre_update_parameters'):
      self.optim.pre_update_parameters()

    for layer in self.ModulesList :
      self.optim.update_params(layer)

    # Check if optimizer has post_update_params method (not all optimizers do)
    if hasattr(self.optim, 'post_update_params'):
      self.optim.post_update_params()

  def train_test_split(self, X, y, test_size=0.2, random_state=None):
    """
    Split data into training and testing sets
    
    Parameters:
    X: input data (features)
    y: target data (labels)
    test_size: fraction of data to use for testing (default 0.2 = 20%)
    random_state: seed for reproducible splits
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    # Create random indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

  def evaluate(self, X_test, y_test):
    """
    Evaluate the model on test data
    
    Parameters:
    X_test: test input data
    y_test: test target data
    
    Returns:
    Dictionary with evaluation metrics
    """
    # Get predictions
    predictions = self.forward(X_test)
    
    # Calculate metrics
    test_loss = self.computeLoss(y_test)
    
    # Calculate MSE and MAE
    mse = np.mean((y_test.flatten() - predictions.flatten()) ** 2)
    mae = np.mean(np.abs(y_test.flatten() - predictions.flatten()))
    
    # Calculate R² score
    y_mean = np.mean(y_test.flatten())
    ss_tot = np.sum((y_test.flatten() - y_mean) ** 2)
    ss_res = np.sum((y_test.flatten() - predictions.flatten()) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return {
        'test_loss': test_loss,
        'mse': mse,
        'mae': mae,
        'r2_score': r2_score,
        'predictions': predictions
    }

