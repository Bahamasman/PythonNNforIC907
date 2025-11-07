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

