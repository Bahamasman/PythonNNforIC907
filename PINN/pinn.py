import sys
import json

#sys.path.append("C:\\Users\\...")

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


############################################# Helper Functions #######################################################
# Numpy array to Torch tensor
def np_to_th(x):
  n_samples = len(x)
  return torch.from_numpy(x).to(torch.float).reshape(n_samples,-1)

# Derivative of outputs w.r.t inputs
def grad(outputs, inputs):
  return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), retain_graph=True, create_graph=True)[0]

def to_float(obj):
  if isinstance(obj, list):
    return [to_float(x) for x in obj]
  else:
    return float(obj)

# Plot loss
def plot_loss(losses):
  plt.figure(figsize=(8,4))
  plt.plot(losses, color='tab:blue')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.grid(True)
  plt.show()

# Plot real solution
def plot_solution(pts:np.ndarray, u:np.ndarray):
  x = pts.T[0]
  t = pts.T[1]
  u = u.ravel()
  # x = pts[:,0]
  # t = pts[:,1]

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
def plot_predictions(net, pts:np.ndarray, u:np.ndarray, pts_train:np.ndarray, u_train:np.ndarray):
  # Predicting displacement using the trained model
  predicted_disp = net.predict(pts)

  x = pts.T[0]
  t = pts.T[1]
  x_train = pts_train.T[0]
  t_train = pts_train.T[1]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, t, u, label='Bar Displacement', color='blue')
  ax.scatter(x_train, t_train, u_train, label='Training Data', color='red')
  ax.plot(x, t, predicted_disp, label='PINN Prediction', color='green')

  ax.set_xlabel('x')
  ax.set_ylabel('t')
  ax.set_zlabel('u(x,t)')
  plt.legend()
  plt.show()

# Plot PINN predictions
def plot_predictions(net, pts:np.ndarray, pts_train:np.ndarray, u_train:np.ndarray):
  # Predicting displacement using the trained model
  predicted_disp = net.predict(pts)

  x = pts.T[0]
  t = pts.T[1]
  x_train = pts_train.T[0]
  t_train = pts_train.T[1]
  u_train = u_train.ravel()
  u_pred = predicted_disp.ravel()
  # x = pts[:,0]
  # t = pts[:,1]

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
  ax.set_zlabel('u_pred')
  plt.legend()
  plt.show()


def density_plot(): #TODO
  fig = plt.figure(figsize=(9, 5)) # Creates a new empty figure (the canvas)
  ax = fig.add_subplot(111) # Adds a single subplot (an Axes object) to the figure, 1 row, 1 column, 1 subplot

  # Displays a 2D array (U_pred.T) as a colored image
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


############################################# Neural Network Classes #######################################################
# Deep Neural Network
class NN(nn.Module):
  def __init__(
      self,
      input_size,
      hidden_size,
      output_size,
      depth,
      layer_list = None,
      act=torch.nn.Tanh):
      super(NN, self).__init__()

      if layer_list:
        self.depth = len(layer_list) - 1
      else:
        self.depth = depth  
      self.activation = act  

      if layer_list:
        layers = list()
        for i in range(self.depth - 1):
          layers.append(('layer_%d' % i, torch.nn.Linear(layer_list[i], layer_list[i+1])))
          layers.append(('activation_%d' % i, self.activation()))
        layers.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layer_list[-2], layer_list[-1])))

      else:
        layers = [('input', torch.nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))
        for i in range(self.depth):
            layers.append(('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size)))
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))  
      
      layerDict = OrderedDict(layers) # built-in dictionary dict that remembers the order in which keys are inserted.
      self.layers = torch.nn.Sequential(layerDict) # sequential composition of other nn.Module objects. Modules are added to nn.Sequential in the order they are passed in the constructor, either as separate arguments or within an OrderedDict.  
  
  def forward(self, x):
      out = self.layers(x)
      return out
  

# Data-driven Solutions
# Data-driven Discovery
class PINN_DynamicBar():
  def __init__(self, X:np.ndarray, u:np.ndarray, NN_infos:list, Xmin, Xmax, layers=None, pde=None, weight_pde=1.0, bc=None, weight_bc=0.0):
      '''Arguments:
      X(nSamples, input_size): inputs [[xi, ti], ...]
      u(nSamples, output_size): outputs [ui, ...]
      NN_infos = input_size, hidden_size, output_size, depth, epochs, learning_rate
      Xmin: bounds of domain
      Xmax: bounds of domain
      '''
      # Boundaries
      self.Xmin = np_to_th(Xmin).to(device)
      self.Xmax = np_to_th(Xmax).to(device)  

      # Data
      self.x = np_to_th(X[:, 0:1]).requires_grad_(True).to(device)
      self.t = np_to_th(X[:, 1:2]).requires_grad_(True).to(device)
      self.u = np_to_th(u).to(device)  

      # Defining learnable parameters
      self.elas = torch.tensor([0.0], requires_grad=True).to(device)
      #self.rho = torch.tensor([0.0], requires_grad=True).to(device)  
      self.elas = torch.nn.Parameter(self.elas)
      #self.rho = torch.nn.Parameter(self.rho)  

      # Deep neural networks
      self.model = NN(NN_infos[0], NN_infos[1], NN_infos[2], NN_infos[3], layers)
      self.pde = pde
      self.bc = bc
      self.weight_pde = weight_pde
      self.weight_bc = weight_bc
      self.model.register_parameter('Elas', self.elas) # explicitly register a torch.nn.Parameter with an nn.Module. This registration ensures that the parameter is included in the module's parameters()
      # self.model.register_parameter('rho', self.rho)  

      # Optimizers
      self.epochs = NN_infos[4]
      self.optim_Adam = torch.optim.Adam(self.model.parameters(), lr = NN_infos[5])
      self.optimizer = torch.optim.LBFGS(
          self.model.parameters(),
          lr=1.0,
          max_iter=50000,
          max_eval=50000,
          history_size=50,
          tolerance_grad=1e-5,
          tolerance_change=1.0 * np.finfo(float).eps,
          line_search_fn="strong_wolfe")  

  def net_u(self, x, t):
      u = self.model(torch.cat([x, t], dim=1)) # concatenate into model inputs
      return u  
  
  # PDE 
  # def net_physics(self, x, t):  
  #     elas = self.elas
  #     # rho = torch.exp(self.rho)
  #     u = self.net_u(x, t)  
  #     dudt = grad(u, t)
  #     d2udt2 = grad(dudt, t)
  #     dudx = grad(u, x)
  #     d2udx2 = grad(elas*dudx, x)   
  #     pde = rho * A * d2udt2 - A * d2udx2 - f
  #     return pde  

  # PDE embeded in the NN through the Loss Function 
  def loss_func(self):
      u_pred = self.net_u(self.x, self.t)
      f_pred = 0
      bc_pred = 0
      if self.bc:
        bc_pred = self.bc(self)
      if self.pde:
        f_pred = self.pde(self, self.x, self.t)
      loss = torch.mean((self.u - u_pred) ** 2) + self.weight_pde*torch.mean(f_pred ** 2) + self.weight_bc*torch.mean(bc_pred ** 2)
      
      self.optimizer.zero_grad()
      loss.backward()  
      self.epochs += 1
      if self.epochs % 100 == 0:
          print('Loss: %e, E: %.5f' %
              (
                  loss.item(),
                  self.elas.item(),
                  # torch.exp(self.rho.detach()).item()
              ))
      return loss  

  
  def train(self):
      self.model.train() # Set model to training mode
      losses = []  
      for epoch in range(self.epochs):
          u_pred = self.net_u(self.x, self.t)
          loss_pde = 0.0
          loss_bc = 0.0
          if self.bc: #! STILL NOT WORKING
            bc_pred = self.bc(self)
            loss_bc = torch.mean(bc_pred ** 2)
          if self.pde:
            pde_pred = self.pde(self, self.x, self.t)
            loss_pde = torch.mean(pde_pred ** 2)
          loss = torch.mean((self.u - u_pred) ** 2) + self.weight_pde*loss_pde + self.weight_bc*loss_bc

          # Backward and optimize
          self.optim_Adam.zero_grad()
          loss.backward()
          self.optim_Adam.step()
          losses.append(loss.item())  
          if epoch % 100 == 0:
              print('It: %d, Loss: %.3e, E: %.5f' %
                  (
                      epoch,
                      loss.item(),
                      self.elas.item(),
                      # torch.exp(self.rho).item()
                  ))  
              
      # Backward and optimize
      # self.optimizer.step(self.loss_func) #using torch.optim.LBFGS

      return losses  
  
  def predict(self, X):
      x = np_to_th(X[:, 0:1]).requires_grad_(True).to(device)
      t = np_to_th(X[:, 1:2]).requires_grad_(True).to(device)  
      self.model.eval() # Set model to evaluation mode
      u = self.net_u(x, t)
      pinn = self.pde(self, self.x, self.t)
      u = u.detach().cpu().numpy()
      pinn = pinn.detach().cpu().numpy()
      return u