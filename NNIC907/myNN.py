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
    
    elif activationFunc == 'LeakReLU':
      #implement me
      raise NameError('Activation_func function' + activationFunc + 'not supported yet!')
    
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
    elif optim == 'AdaGrad':
      self.optim = Optimizer_AdaGrad(learningRate,decay=1e-5)
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

