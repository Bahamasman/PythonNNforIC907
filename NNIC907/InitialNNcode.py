import numpy as np
import matplotlib.pyplot as plt

import myNN as nn

#np.random.seed(0) #Setting seed = 0 for debugging

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

hidden_size = 10
num_hiddenLayers = 2
input_size = 1
output_size = 1
rate = 1.0 
activationf = 'ReLU'

NeuralNetwork = nn.myNeuralNetwork()
NeuralNetwork.build(input_size,output_size,num_hiddenLayers,hidden_size,activationf,rate)
NeuralNetwork.intialize()


NeuralNetwork.definieLossFunction() #Update me
NeuralNetwork.defineOptimizer(lr) #Update me

# Convert data to our data structure
# Implement using reshape
xptsbatch = xpts.reshape(npts,1)
yrealbatch = yreal.reshape(npts,1)

# Training loop
for epoch in range(epochs):
  
  # Forward pass
  out = NeuralNetwork.forward(xptsbatch)  

  # Compute loss (mean squared error)
  lossk = NeuralNetwork.computeLoss(yrealbatch)

  # Compute the derivatives
  NeuralNetwork.backward(yrealbatch)

  # Gradient descent step
  NeuralNetwork.updateParam()

  # Store loss and parameters for plotting
  lossvec[epoch] = lossk

  # Print every 500 epochs
  if epoch % 500 == 0:
    print(f"epoch {epoch}: loss = {lossk:.10f}")

# Create a 1x2 subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Left: Data and model fit
axs[0].plot(xpts, yreal, 'o', label='true data')
axs[0].plot(xpts, NeuralNetwork.output, '-', label='NN prediction')
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
