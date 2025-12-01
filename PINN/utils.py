import sys
import json

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# import traceback
from collections import OrderedDict
# from tqdm import tqdm

# For reproducibility
torch.manual_seed(0)
np.random.seed(0)

############################################# Helper Functions #######################################################
# Numpy array to Torch tensor
def np_to_th(x):
  n_samples = len(x)
  return torch.from_numpy(x).to(torch.float).reshape(n_samples,-1)

# Derivative of outputs w.r.t inputs
def grad(outputs, inputs):
    """
    returns gradients d(outputs)/d(inputs) with shape like inputs.
    If grad is None (unused) returns zeros_like(inputs).
    """
    if not (isinstance(outputs, torch.Tensor) and isinstance(inputs, torch.Tensor)):
        raise ValueError("safe_grad expects torch.Tensor for outputs and inputs")
    grads = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                                retain_graph=True, create_graph=True, allow_unused=True)
    g = grads[0]
    if g is None:
        return torch.zeros_like(inputs)
    return g

def to_float(obj):
  if isinstance(obj, list):
    return [to_float(x) for x in obj]
  else:
    return float(obj)
  
def is_numeric(s):
    try:
        float(s)  # Try converting to a float 
        return True # Return True
    except Exception:
        return False # Return False otherwise

# Plot loss
def plot_loss(losses, title='Training Total Loss'):
  plt.figure(figsize=(8,4))
  plt.plot(losses, color='tab:blue')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(title)
  plt.grid(True)
  plt.show()

# Plot real solution
def plot_solution(pts:np.ndarray, u:np.ndarray):
  x = pts.T[0]
  t = pts.T[1]
  u = u.ravel()

  xi = np.linspace(x.min(), x.max(), 1000)
  ti = np.linspace(t.min(), t.max(), 1000)
  XI, TI = np.meshgrid(xi, ti)
  UI = griddata(pts, u, (XI, TI), method='cubic')
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(XI, TI, UI, label='Dynamic Bar', color='blue')
  ax.set_xlabel('x')
  ax.set_ylabel('t')
  ax.set_zlabel('u(x,t)')
  plt.show()

# Plot PINN predictions
def plot_predictions(net:PINN_DynamicBar, pts:np.ndarray, pts_train:np.ndarray, u_train:np.ndarray):
  # Predicting displacement using the trained model
  predicted_disp = net.predict(pts)

  x = pts.T[0]
  t = pts.T[1]
  x_train = pts_train.T[0] 
  t_train = pts_train.T[1]
  u_train = u_train.ravel()
  u_pred = predicted_disp.ravel()

  # Create a regular grid
  xi = np.linspace(x.min(), x.max(), 1000)
  ti = np.linspace(t.min(), t.max(), 1000)
  XI, TI = np.meshgrid(xi, ti)

  UI = griddata(pts, u_pred, (XI, TI), method='cubic')

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(XI, TI, UI, label='PINN Prediction', color='green')

  ax.set_xlabel('x')
  ax.set_ylabel('t')
  ax.set_zlabel('u pred')
  plt.legend()
  plt.show()

def plot_prediction_E(net:PINN_DynamicBar, X:np.ndarray, E_exact:np.ndarray, E_pred:np.ndarray):
  plt.figure(figsize=(8,4))
  plt.plot(X[:,0:1], E_exact, 'b-', label='Exact E(x)')
  plt.plot(X[:,0:1], E_pred, 'r--', label='PINN Predicted E(x)')
  plt.xlabel('x')
  plt.ylabel('E(x)')
  plt.title('Exact vs Predicted E(x)')
  plt.grid(True)
  plt.show()


def density_plot(): #TODO
  fig = plt.figure(figsize=(9, 5)) # Creates a new empty figure (the canvas)
  ax = fig.add_subplot(111) # Adds a single subplot (an Axes object) to the figure, 1 row, 1 column, 1 subplot

  # Displays a 2D array (U_pred.T) as a colored image
  UI = griddata(X_star, u_star, (X, T), method='cubic') #! APAGAR
  UI = griddata(pts, u_pred, (XI, TI), method='cubic')
  h = ax.imshow(UI, interpolation='nearest', cmap='rainbow', 
              extent=[x.min(), x.max(), t.min(), t.max()],
              origin='lower', aspect='auto')

  # Colorbar axes configs
  divider = make_axes_locatable(ax) # “attach” new axes next to an existing one (used for colorbars).
  cax = divider.append_axes("right", size="5%", pad=0.10) # Creates a new vertical axis to the right side, with spacing pad
  cbar = fig.colorbar(h, cax=cax) # Creates a colorbar showing the mapping between colors and values.
  cbar.ax.tick_params(labelsize=15)

  # Plot training data points
  ax.plot(X_train[:,0], X_train[:,1], 'kx', label = f'Data ({nSamples} points)',
        markersize = 4, clip_on = False, alpha=.5)

  # Plot vertical white lines at selected times
  # line = np.linspace(x.min(), x.max(), 2)[:,None]
  # ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
  # ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
  # ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

  ax.set_xlabel('$x$', size=20)
  ax.set_ylabel('$t$', size=20)
  ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
  ax.set_title('$u(x,t)$', fontsize = 20)
  ax.tick_params(labelsize=15)
  plt.show()


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')