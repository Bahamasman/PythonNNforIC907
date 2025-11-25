import numpy as np
import matplotlib.pyplot as plt

import myNN as nn

# Define the ill-conditioned quadratic function: f(x,y) = 100*x^2 + y^2
def f(x):
    # x is a 2D array-like: [x, y]
    return 100 * x[0]**2 + x[1]**2

npts = 50
xpts = np.linspace(-5, 5, npts)
ypts = np.linspace(-5, 5, npts)

# Generating data (grid)
freal = np.array([[f([x, y]) for x in xpts] for y in ypts])  # shape (npts, npts)

# Learning rate and number of epochs
lr = 0.1
epochs = 20001  # Reduced for testing
lossvec1 = np.zeros(epochs)
lossvec2 = np.zeros(epochs)

# Network configuration
hidden_size = [20,20]
num_hiddenLayers = len(hidden_size)
input_size = 2
output_size = 1
rate = 1.0
activationf = 'LeakyReLU'
optim1 = 'SGD_Decay'
optim2 = 'Adam'
# Convert 2D grid to training batches: inputs shape (N,2), targets shape (N,1)
xv, yv = np.meshgrid(xpts, ypts)                      # shape (npts, npts)
xptsbatch = np.column_stack((xv.ravel(), yv.ravel()))  # (npts*npts, 2)
frealbatch = freal.ravel().reshape(-1, 1)             # (npts*npts, 1)

# Normalize the data to prevent gradient explosion
freal_mean = frealbatch.mean()
freal_std = frealbatch.std()
frealbatch_norm = (frealbatch - freal_mean) / freal_std

print(f"Original data range: {frealbatch.min():.2f} to {frealbatch.max():.2f}")
print(f"Normalized data range: {frealbatch_norm.min():.2f} to {frealbatch_norm.max():.2f}")

# Use normalized data for training
training_targets = frealbatch_norm

# Training loop
NeuralNetwork1 = nn.myNeuralNetwork()
NeuralNetwork1.build(input_size, output_size, num_hiddenLayers, hidden_size, activationf, rate)
NeuralNetwork1.initialize()


NeuralNetwork1.defineLossFunction()
print("Training started for SGD with Decay...")
NeuralNetwork1.defineOptimizer(optim1, lr)
for epoch in range(epochs):
    # Forward pass (some implementations store outputs internally; forward may return values)
    out = NeuralNetwork1.forward(xptsbatch)

    # Compute loss (mean squared error)
    lossk = NeuralNetwork1.computeLoss(training_targets)

    # Backprop and update
    NeuralNetwork1.backward(training_targets)
    NeuralNetwork1.updateParam()

    lossvec1[epoch] = lossk

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss = {lossk:.10f}")

# After training: get predictions on the grid
preds_norm_SDG = NeuralNetwork1.forward(xptsbatch)  # expected shape (N,1)

# Denormalize predictions back to original scale
preds_SDG = preds_norm_SDG * freal_std + freal_mean
preds_grid_SDG = preds_SDG.reshape(npts, npts)

# Training loop
NeuralNetwork2 = nn.myNeuralNetwork()
NeuralNetwork2.build(input_size, output_size, num_hiddenLayers, hidden_size, activationf, rate)
NeuralNetwork2.initialize()
NeuralNetwork2.defineLossFunction()
print("Training started for Adam...")
NeuralNetwork2.defineOptimizer(optim2, lr)          
for epoch in range(epochs):
    # Forward pass (some implementations store outputs internally; forward may return values)
    out = NeuralNetwork2.forward(xptsbatch)

    # Compute loss (mean squared error)
    lossk = NeuralNetwork2.computeLoss(training_targets)

    # Backprop and update
    NeuralNetwork2.backward(training_targets)
    NeuralNetwork2.updateParam()

    lossvec2[epoch] = lossk

    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss = {lossk:.10f}")

# After training: get predictions on the grid
preds_norm_ADAM = NeuralNetwork2.forward(xptsbatch)  # expected shape (N,1)

# Denormalize predictions back to original scale
preds_ADAM = preds_norm_ADAM * freal_std + freal_mean
preds_grid_ADAM = preds_ADAM.reshape(npts, npts)

# Plot true function and model prediction as heatmaps, and loss curve
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# Left: true vs predicted as two-pane image (use imshow)
im0 = axs[0].imshow(freal, extent=[xpts.min(), xpts.max(), ypts.min(), ypts.max()],
                    origin='lower', aspect='auto')
axs[0].set_title('True function (f)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(preds_grid_SDG, extent=[xpts.min(), xpts.max(), ypts.min(), ypts.max()],
                    origin='lower', aspect='auto')
axs[1].set_title('NN prediction (SGD)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(preds_grid_ADAM, extent=[xpts.min(), xpts.max(), ypts.min(), ypts.max()],
                    origin='lower', aspect='auto')
axs[2].set_title('NN prediction (Adam)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
fig.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.show()

# Plot loss in a separate figure for clarity
plt.figure(figsize=(6, 4))
plt.plot(lossvec1, label='Loss (SGD)')
plt.plot(lossvec2, label='Loss (Adam)',alpha=0.7)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.yscale('log')  # optional: log scale often helpful
plt.legend()
plt.show()
