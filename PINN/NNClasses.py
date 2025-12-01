from Utils import *
from ScalesClass import Scales

############################################# Neural Network Classe #######################################################
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
  

############################################# PINN Classe #######################################################
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
         self.Elas = self.build_E().to(device)
         self.net_params = list(self.model.parameters()) + list(self.Elas.parameters())

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
      x_scaled = self.scales.x_phys_to_scaled(x)
      t_scaled = self.scales.t_phys_to_scaled(t)
      u = self.model(torch.cat([x_scaled, t_scaled], dim=1)) # concatenate into model inputs
      return u  

  # E Network output E_scaled
  def net_E(self, x):  
    # convert physical -> scaled
    x_scaled = self.scales.x_phys_to_scaled(x)
    E_activation = nn.Softplus() # soft activation function, ensures E > 0 and a learnable parameter of the Network
    # If E is parameter (scalar) we shouldn't call this; when E is a learnable network use `self.E_net`.
    if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
      raw = self.Elas(x_scaled)
      return E_activation(raw) + 1e-8
    else:
      # In the scalar-Elas case, return the scalar parameter (broadcast to match x shape if needed)
      return self.Elas * torch.ones_like(x)
  
  # PDE residual in adimensional variables
  def net_physics(self, x, t): 
    """
    Compute PDE residual in scaled variables so that the nondimensional PDE is:
        u_tt_tilde - (E_tilde * u_x_tilde)_x_tilde = f_tilde

    We compute derivatives with respect to the *scaled* variables x_tilde, t_tilde.
    """
    # u_scaled = self.net_u(x, t) 
    # E_scaled = self.net_E(x)

    # convert physical -> scaled
    x_scaled = self.scales.x_phys_to_scaled(x)
    t_scaled = self.scales.t_phys_to_scaled(t)

    if not x_scaled.requires_grad:
      x_scaled = x_scaled.clone().detach().requires_grad_(True)
    if not t_scaled.requires_grad:
      t_scaled = t_scaled.clone().detach().requires_grad_(True)

    inp = torch.cat([x_scaled, t_scaled], dim=1)
    u_scaled = self.model(inp)   # ũ(x̃,t̃)
    if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
      raw = self.Elas(x_scaled)   # network expects scaled x
      E_scaled = F.softplus(raw) + 1e-8
    else:
       E_scaled = self.Elas * torch.ones_like(x_scaled)
     
    dudt = grad(u_scaled, t_scaled)
    d2udt2 = grad(dudt, t_scaled)
    dudx = grad(u_scaled, x_scaled)
    d2udx2 = grad(dudx, x_scaled) 
    dEdx = grad(E_scaled, x_scaled)
  
    # Evaluate f_scaled: use f_th_func (expects physical x) and F0 computed in Scales
    try:
      f_phys = self.scales.f_th_func(x)
      if isinstance(f_phys, torch.Tensor):
        f_scaled = f_phys / self.scales.F0
      else:
        f_scaled = np_to_th(f_phys) / self.scales.F0
    except Exception:
      raise Exception("f(x) expects a tensor type input")
     
    # Compute the nondimensional PDE residual:
    pde_res = d2udt2 - (E_scaled * d2udx2 + dEdx * dudx) - f_scaled
    return pde_res
     
  
  def compute_losses(self, nCollocation): 
      """
      Compute data loss, PDE loss and BC loss with consistent collocation sampling.
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
      loss_pde = torch.tensor(0.0).to(device)
      if self.pde is not None:
        pde_res = self.net_physics(x_colloc, t_colloc)  
        loss_pde = torch.mean(pde_res**2)

      # BC loss
      loss_bc = torch.tensor(0.0).to(device)
      if self.bc is not None:
        bc_res = self.bc(self)
        loss_bc = torch.mean(bc_res**2)

      total = loss_data + self.weight_pde * loss_pde + self.weight_bc * loss_bc

      return total, loss_data, loss_pde, loss_bc
     
  
  def train(self, nCollocation):
      self.model.train() # Set model to training mode
      if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
        self.Elas.train()
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
          if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
            x_mid = torch.tensor([[0.5*self.scales.L]], dtype=torch.float32, device=device)
            E_mid = self.Elas(x_mid).detach().cpu().numpy().squeeze() 
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
      if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
        self.Elas.eval()

      with torch.no_grad():
        u_scaled = self.net_u(x, t)
      u_phys = self.scales.scaled_to_u_phys(u_scaled.detach().cpu().numpy())
      return u_phys

  def predict_E(self, x): 
    # x: numpy array of physical x values shape (N,1) or (N,)
    x_th = np_to_th(x).to(device).reshape(-1,1).requires_grad_(False)
    # convert to scaled x for network evaluation
    x_scaled = self.scales.x_phys_to_scaled(x_th)

    if hasattr(self, 'Elas') and isinstance(self.Elas, nn.Module):
      self.Elas.eval()
      with torch.no_grad():
         E_scaled = self.Elas(x_scaled)
         E_scaled = F.softplus(E_scaled) + 1e-8
    else:
      with torch.no_grad():
         E_scaled = self.Elas * torch.ones_like(x_scaled)
    E_phys = self.scales.scaled_to_E_phys(E_scaled.detach().cpu().numpy())
    return E_phys.reshape(-1,1)