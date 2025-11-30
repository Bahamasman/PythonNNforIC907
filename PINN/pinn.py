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
def plot_predictions(net, pts:np.ndarray, pts_train:np.ndarray, u_train:np.ndarray):
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
  ax.set_zlabel('u_pred')
  plt.legend()
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
  
class Scales: #! ENTENDER
    def __init__(self, L0, E0, A0, rho0, f0_np, f0_th):
        self.L = float(L0)
        self.E0 = float(E0)
        self.A = float(A0)
        self.rho = float(rho0)
        self.f_func = f0_np   
        self.f_th_func = f0_th

        # Compute F0 = max |f(x)| for x ∈ [0, L]
        if callable(self.f_func):
            xs = np.linspace(0.0, self.L, 2001)
            fvals = self.f_func(xs)

            if isinstance(fvals, np.ndarray): 
                self.F0 = float(np.max(np.abs(fvals)) + 1e-12)
            else:
                # In case the f_func returns a tensor
                try:
                    self.F0 = float(torch.max(torch.abs(torch.tensor(fvals))) + 1e-12)
                except:
                    self.F0 = float(np.max(np.abs(np.array(fvals))) + 1e-12)
        else:
            # Case f_numpy is constant
            self.F0 = float(abs(f0_np) + 1e-12)

        # Time scale: T = L * sqrt(rho / E)
        self.T = float(np.sqrt(self.rho * self.L**2 / self.E0))

        # Displacement scale so the forcing coefficient becomes O(1)
        # U = F0 * L^2 / (E0 * A)
        self.U = float(self.F0 * self.L**2 / (self.E0 * self.A))

    # ==========================================================
    #   CONVERSION FUNCTIONS: PHYS → SCALED AND SCALED → PHYS
    # ==========================================================
    def x_phys_to_scaled(self, x_phys):
        return x_phys / self.L

    def t_phys_to_scaled(self, t_phys):
        return t_phys / self.T

    def u_phys_to_scaled(self, u_phys):
        return u_phys / self.U

    def E_phys_to_scaled(self, E_phys):
        return E_phys / self.E0

    def f_phys_to_scaled(self, x_phys):
        """
        Return f_scaled(x) = f(x) / F0
        x_phys can be np.array or torch.tensor
        """
        try:
            fvals = self.f_func(x_phys)

            if isinstance(fvals, torch.Tensor):
                return fvals / self.F0
            else:
                return np.array(fvals) / self.F0

        except:
            return np.array(self.f_func(x_phys)) / self.F0

    def scaled_to_u_phys(self, u_scaled):
        return u_scaled * self.U

    def scaled_to_E_phys(self, E_scaled):
        return E_scaled * self.E0

      
# Data-driven Solutions
# Data-driven Discovery
class PINN_DynamicBar():
  def __init__(self, X:np.ndarray, u:np.ndarray, NN_infos:list, Xmin, Xmax, scales:Scales, layers=None, loss_f=nn.MSELoss(), pde=None, weight_pde=1.0, bc=None, weight_bc=0.0):
      '''Arguments:
      X(nSamples, input_size): inputs [[xi, ti], ...]
      u(nSamples, output_size): outputs [ui, ...]
      NN_infos = input_size, hidden_size, output_size, depth, epochs, learning_rate
      Xmin: bounds of domain
      Xmax: bounds of domain
      '''
      self.scales = scales
      
      # Boundaries
      self.Xmin = np_to_th(Xmin).to(device)
      self.Xmax = np_to_th(Xmax).to(device)  
      self.x_min, self.t_min = self.Xmin[0], self.Xmin[1]
      self.x_max, self.t_max = self.Xmax[0], self.Xmax[1]

      # Data
      self.x = np_to_th(X[:, 0:1]).requires_grad_(True).to(device)
      self.t = np_to_th(X[:, 1:2]).requires_grad_(True).to(device)
      # self.u = np_to_th(u).to(device)  
      # u provided in physical units -> convert to u_scaled for loss comparisons
      self.u_phys = np_to_th(u).to(device)
      self.u = self.scales.u_phys_to_scaled(self.u_phys).to(device)  # u_scaled targets

      # Deep neural networks for u_scaled
      self.model = NN(NN_infos[0], NN_infos[1], NN_infos[2], NN_infos[3], layers)
      self.pde = pde
      if self.pde is not None:
        self.A, self.rho, self.E_type, self.f_func = self.pde()
      self.bc = bc
      self.loss_f = loss_f
      self.weight_pde = weight_pde
      self.weight_bc = weight_bc

      # Defining learnable parameters
      if hasattr(self, 'E_type') and self.E_type == "Polynomial":
         # E(x) is MLP
         self.E_net = self.build_E().to(device)
         self.net_params = list(self.model.parameters()) + list(self.E_net.parameters())

      else:
         # E is scalar
         self.Elas = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=True))
         # self.model.register_parameter("Elas", self.Elas) #! VERIFICAR SE ISSO É NECESSÁRIO,
         self.net_params = list(self.model.parameters()) + [self.Elas]

      # Optimizers
      self.epochs = NN_infos[4]
      self.optim_Adam = torch.optim.Adam(self.net_params, lr = NN_infos[5])
      self.optim_LBFGS = torch.optim.LBFGS(
          self.net_params,
          lr=1.0,
          max_iter=5000,
          max_eval=50000,
          history_size=50,
          tolerance_grad=1e-9,
          tolerance_change=1e-12, 
          line_search_fn="strong_wolfe") 
     
  def build_E(self): # E Network (MLP) that gives as output E_scaled
    return nn.Sequential(
           nn.Linear(1, 32),
           nn.Tanh(),
           nn.Linear(32, 32),
           nn.Tanh(),
           nn.Linear(32, 1))
  
  # Network output u_scaled 
  def net_u(self, x, t):
      # convert physical -> scaled
      x_scaled = self.scales.x_phys_to_scaled(x)
      t_scaled = self.scales.t_phys_to_scaled(t)
      # x_n = 2.0*(x - self.x_min)/(self.x_max - self.x_min) - 1.0 #! SEE THIS
      # t_n = 2.0*(t - self.t_min)/(self.t_max - self.t_min) - 1.0
      u = self.model(torch.cat([x_scaled, t_scaled], dim=1)) # concatenate into model inputs
      return u  

  # E Network output E_scaled
  def net_E(self, x):  # x_tilde: torch tensor shape (N,1)
    x_scaled = self.scales.x_phys_to_scaled(x)
    E_activation = nn.Softplus() # soft activation function, ensures E > 0 and a learnable parameter of the Network
    # If E is parameter (scalar) we shouldn't call this; when E is a learnable network use `self.E_net`.
    if hasattr(self, 'E_net') and isinstance(self.E_net, nn.Module):
      raw = self.E_net(x_scaled)
      return E_activation(raw) + 1e-8
    else:
      # In the scalar-Elas case, return the scalar parameter (broadcast to match x shape if needed)
      return self.Elas * torch.ones_like(x)
  
  # PDE residual in adimensional variables
  def net_physics(self, x, t): 
    u_scaled = self.net_u(x, t) # returns u_scaled
    if isinstance(self.Elas, nn.Module):
       E_scaled = self.net_E(x)
    else:
       E_scaled = self.Elas * torch.ones_like(x)
     
    dudt = grad(u_scaled, t)
    d2udt2 = grad(dudt, t)
    dudx = grad(u_scaled, x)
    d2udx2 = grad(E_scaled*dudx, x) 

    # Convert physical derivatives -> tilde derivatives:
    d2udt2_scaled = (self.scales.T**2) * d2udt2 # d^2/dt_scaled^2 u_scaled = T^2 * d^2/dt_phys^2 u_scaled
    d2udx2_scaled = (self.scales.L**2) * d2udx2 # d^2/dx_scaled^2 u_scaled = T^2 * d^2/dx_phys^2 u_scaled
  
    try:
      f_phys = self.scales.f_th_func(x)
      if isinstance(f_phys, torch.Tensor):
        f_scaled = f_phys / self.scales.F0
    except Exception:
      raise Exception("f(x) expects a tensor type input")
     
    pde_res = d2udt2_scaled - d2udx2_scaled - f_scaled
    return pde_res
     
  
  def compute_losses(self, nCollocation): 
      """
      x_colloc, t_colloc: PDE residual evaluation
      u_bc: boundary condition points
      """

      # Pre-generate uniform collocation points
      x_min_f = float(self.x_min.detach().cpu().item())
      x_max_f = float(self.x_max.detach().cpu().item())
      t_min_f = float(self.t_min.detach().cpu().item())
      t_max_f = float(self.t_max.detach().cpu().item())
      x_colloc_np = np.random.uniform(x_min_f, x_max_f, size=(nCollocation, 1))
      t_colloc_np = np.random.uniform(t_min_f, t_max_f, size=(nCollocation, 1))
      x_colloc = np_to_th(x_colloc_np).requires_grad_(True).to(device)
      t_colloc = np_to_th(t_colloc_np).requires_grad_(True).to(device)

      # Data loss (observations)
      u_pred_scaled = self.net_u(self.x, self.t)
      #loss_data = torch.mean((self.u - u_pred_data)**2)
      loss_data = self.loss_f(u_pred_scaled, self.u)

      # PDE loss
      loss_pde = torch.tensor(0.0, requires_grad=True).to(device)
      if self.pde is not None:
        pinn = self.net_physics(x_colloc, t_colloc)  
        loss_pde = torch.mean(pinn**2)

      # BC loss
      loss_bc = torch.tensor(0.0).to(device)
      if self.bc is not None:
        bc_res = self.bc(self)
        loss_bc = torch.mean(bc_res**2)

      total = loss_data + self.weight_pde * loss_pde + self.weight_bc * loss_bc

      return total, loss_data, loss_pde, loss_bc
     
  
  def train(self, nCollocation):
      self.model.train() # Set model to training mode
      if hasattr(self, 'E_net') and isinstance(self.E_net, nn.Module):
        self.E_net.train()
      losses = []  
      losses_Data = []
      losses_PDE = []
      losses_BC = []
      
      for epoch in range(self.epochs):
          
        # Compute Losses
        total_loss, lossData, lossPDE, lossBC = self.compute_losses(nCollocation)

        # Optimizer and Backward
        self.optim_Adam.zero_grad()
        total_loss.backward()
        self.optim_Adam.step()

        losses.append(total_loss.item()) 
        losses_Data.append(lossData.item())
        losses_BC.append(lossBC.item())
        losses_PDE.append(lossPDE.item())

        if epoch % 100 == 0:
          # compute representative E value at mid domain
          if hasattr(self, 'E_net') and isinstance(self.E_net, nn.Module):
            x_mid = torch.tensor([[0.5*self.scales.L]], dtype=torch.float32, device=device)
            E_mid = self.net_E(x_mid).detach().cpu().numpy().squeeze() 
            E_phys = self.scales.scaled_to_E_phys(E_mid)
          else:
            E_phys = self.scales.scaled_to_E_phys(self.Elas.item())
          print(f"Epoch {epoch}/{self.epochs}:")
          print(f"Loss Data {lossData.item():.3e} | Loss PDE: {lossPDE.item():.3e} | Loss BC: {lossBC:.5f}")
          print(f"Total Loss: {total_loss.item():.3e} | E: {E_phys:.5f}")
              
      # captures collocation points
      def loss_func():
        self.optim_LBFGS.zero_grad()
        total_loss, *_ = self.compute_losses(nCollocation)
        total_loss.backward()
        return total_loss
      
      print("Refinament with L-BFGS...")
      self.optim_LBFGS.step(loss_func) 

      return losses, losses_Data, losses_PDE, losses_BC 
  
  def predict(self, X):
      x = np_to_th(X[:, 0:1]).requires_grad_(True).to(device)
      t = np_to_th(X[:, 1:2]).requires_grad_(True).to(device)  

      self.model.eval() # Set to Evaluation mode
      if hasattr(self, 'E_net') and isinstance(self.E_net, nn.Module):
        self.E_net.eval()

      with torch.no_grad():
        u_scaled = self.net_u(x, t)
      u_phys = self.scales.scaled_to_u_phys(u_scaled.detach().cpu().numpy())
      return u_phys

  def predict_E(self, x): 
    x_th = np_to_th(x).requires_grad_(True).to(device)
    # x_th = torch.tensor(x, dtype=torch.float32, device=device).view(-1,1) 

    if hasattr(self, 'E_net') and isinstance(self.E_net, nn.Module):
      self.E_net.eval()
      with torch.no_grad():
         E_scaled = self.net_E(x_th)
    
    else:
      with torch.no_grad():
         E_scaled = self.Elas * torch.ones_like(x_th)

    E_phys = self.scales.scaled_to_E_phys(E_scaled.detach().cpu().numpy())
    return E_phys

  
