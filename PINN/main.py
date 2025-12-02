from NNClasses import *
import time
import os

#TODO Adicionar ruído nos dados
#TODO Escolher 4 casos diferentes de E(x) e f(x) (constante, polinomial, piecewise, outro)

############################################# Impoting Data #######################################################
input_path = "C:\\Users\\theyd\\OneDrive\\Desktop\\marina\\PythonNNforIC907\\PINN\\InputData\\"
input_file = "data2.json"
case = "2"
input_data = os.path.join(input_path, input_file)

with open(input_data) as myFile:
    mathematica_data = json.load(myFile)

# physics informed neural network f(x, t)
L = to_float(mathematica_data["Properties"]["L"])
Interval = to_float(mathematica_data["Properties"]["Interval"])
A = to_float(mathematica_data["Properties"]["A"])
rho = to_float(mathematica_data["Properties"]["rho"])
E_type = mathematica_data["Properties"]["E"]
f_type = mathematica_data["Properties"]["f"]

if("u_x0") in mathematica_data["BCs_ICs"]: u_x0 = to_float(mathematica_data["BCs_ICs"]["u_x0"])
else: u_x0 = None
if("u_xL") in mathematica_data["BCs_ICs"]: u_xL = to_float(mathematica_data["BCs_ICs"]["u_xL"])
else: u_xL = None
if("u_t0") in mathematica_data["BCs_ICs"]: u_t0 = to_float(mathematica_data["BCs_ICs"]["u_t0"])
else: u_t0 = None
if("du_dx0") in mathematica_data["BCs_ICs"]: du_dx0 = to_float(mathematica_data["BCs_ICs"]["du_dx0"])
else: du_dx0 = None

def E(x:torch.Tensor):
  if is_numeric(E_type):
    return to_float(E_type)*torch.ones_like(x)
  elif E_type == "Polynomial":
    return 2*x
  elif E_type == "Piecewise":
    return torch.where(x < 2., 20e6, 1000000) 


def f(x:torch.Tensor):
  if is_numeric(f_type):
    return to_float(f_type)*torch.ones_like(x)
  elif f_type == "Polynomial":
    return -200*x
  elif f_type == "Piecewise":
    return torch.where(x < 2., torch.tensor(10000000.0), torch.tensor(0.0)) # allows for element-wise selection from two tensors based on a boolean condition

def f_numpy(x:np.ndarray):
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    return f(x_t).detach().cpu().numpy().flatten()

############################################# Initial Configurations #######################################################
# Dynamic-Bar PDE Parameters
def physics_infos(): 
  return A, rho, E_type, f_type

#Boundary/Initial Conditions
def net_bc(model:PINN_DynamicBar):
    bc_res_list = []
    n_bc = 300

    t0 = torch.linspace(0, Interval, n_bc, device=device).view(-1, 1).requires_grad_(True) # time points for BCs

    # left boundary u_x0
    if u_x0 is not None:
        x0 = torch.zeros_like(t0).requires_grad_(True)
        u_x0_pred_scaled = model.net_u(x0, t0)
        u_x0_scaled = torch.tensor(model.scales.u_phys_to_scaled(u_x0), dtype=torch.float32, device=device).unsqueeze(-1)
        bc_res_list.append(u_x0_pred_scaled - u_x0_scaled)

    # right boundary u_xL
    if u_xL is not None:
        xL = L * torch.ones_like(t0).requires_grad_(True)
        u_xL_pred_scaled = model.net_u(xL, t0)
        u_xL_scaled = torch.tensor(model.scales.u_phys_to_scaled(u_xL), dtype=torch.float32, device=device).unsqueeze(-1)
        bc_res_list.append(u_xL_pred_scaled - u_xL_scaled)

    # initial condition u_t0 
    if u_t0 is not None:
        x_init = torch.linspace(0, L, n_bc, device=device).view(-1, 1).requires_grad_(True) # spatial points for ICs
        t_init = torch.zeros_like(x_init)
        u_t0_pred = model.net_u(x_init, t_init)
        u_t0_scaled = torch.tensor(model.scales.u_phys_to_scaled(u_t0), dtype=torch.float32, device=device).unsqueeze(-1)
        bc_res_list.append(u_t0_pred - u_t0_scaled)

    if len(bc_res_list) == 0:
        # No BCs provided: return zero residual (so BC loss = 0)
        return torch.zeros((1,1), device=device)
    else:
        return torch.cat(bc_res_list, dim=0)
    
# Exact solution data set
x = np.array(to_float(mathematica_data["x"])).flatten()[:,None]
t = np.array(to_float(mathematica_data["t"])).flatten()[:,None]
u = np.array(to_float(mathematica_data["u"])).T

X, T = np.meshgrid(x,t)
# Given two one-dimensional arrays, x and y, np.meshgrid(x, y) returns two 2-D arrays, X and Y.
# The array X contains the x-coordinates of all points on the grid, with x values repeated row-wise.
# The array Y contains the y-coordinates of all points on the grid, with y values repeated column-wise.

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # stack arrays horizontally (into columns)
u_star = u.flatten()[:,None]

# Domain bounds
Xmin = X_star.min(0) 
Xmax = X_star.max(0)

############################################# Training on Noisy Data #######################################################
def set_up_training_set(noise_level:float):
  plot_solution(X_star, u_star, case)
  # Training data set
  nSamples = 5000
  noise = noise_level # noise level for u data
  sample = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
  X_train = X_star[sample,:]
  u_train = u_star[sample,:] + np.random.uniform(-noise,noise,nSamples) # Adding some noise
  # u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
  density_plot(X_star, u_star, X_train, case) # Plotting the training data points

  return X_train, u_train

def Simulation(nLayers:int, nNeurons:int, nEpochs:int, nCollocations:int, X_train, u_train, learning_rate = 1e-3, iter="0"):
   # Neural Network Architecture
   NN_infos = [2, nNeurons, 1, nLayers, nEpochs, learning_rate] # input_size, hidden_size, output_size, depth, epochs, learning_rate
   #layers = [2, 30, 30, 30, 30, 1]

   # Create Scales object 
   scales = Scales(L0=L, E0=E, A0=A, rho0=rho, f0_np=f_numpy, f0_th=f) 
   
   # print(f"Neural Network Info: \n\t Number of Neurons: {width} \n\t Number of Layers: {depth} \n\t Epochs: {epochs} \n\t Learning Rate: {lr}")

   # Training
   model = PINN_DynamicBar(X_train, u_train, NN_infos, Xmin, Xmax, scales, pde=physics_infos, bc=net_bc, weight_pde=2.0, weight_bc=2.0)
   initial_time = time.time()
   losses, losses_Data, losses_PDE, losses_BC = model.train(nCollocations)
   end_time = time.time()
   total_time = end_time - initial_time

   # print(f"Training Time: {total_time:.3f} s")

   plot_loss(losses, iter, case)
   # plot_loss(losses_PDE, title="PDE Loss")
   # plot_loss(losses_Data, title="Data Loss")
   # plot_loss(losses_BC, title="BC Loss")

   # Prediction (physical)
   u_pred = model.predict(X_star)
   U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
   error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
   #print(f"Relative L2 error (u): {error_u:.3e}")

   # Estimate E(x) (physical) 
   E_pred = model.predict_E(X_star[:,0:1])
   error_E, E_real_np, E_pred_np = relative_error_E(E, E_pred, X_star[:,0:1])

   # print(f"Relative L2 error (E): {error_E:.3e}")
   plot_prediction_E(model, X_star, E_real_np, E_pred_np, iter, case)
   plot_predictions(model, X_star, X_train, u_star, iter, case)
  
   return total_time, error_u, error_E

############################################# Plotting Results #######################################################
# fig = plt.figure(figsize=(9, 10)) # Creates a new empty figure (the canvas)
# ax = fig.add_subplot(111) # Adds a single subplot (an Axes object) to the figure, 1 row, 1 column, 1 subplot

# # Displays a 2D array (U_pred.T) as a colored image
# h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
#               extent=[x.min(), x.max(), t.min(), t.max()],
#               origin='lower', aspect='auto')

# # Colorbar axes configs
# divider = make_axes_locatable(ax) # “attach” new axes next to an existing one (used for colorbars).
# cax = divider.append_axes("right", size="5%", pad=0.10) # Creates a new vertical axis to the right side, with spacing pad
# cbar = fig.colorbar(h, cax=cax) # Creates a colorbar showing the mapping between colors and values.
# cbar.ax.tick_params(labelsize=15)

# # Plot training data points
# ax.plot(X_train[:,0], X_train[:,1], 'kx', label = f'Data ({nSamples} points)',
#         markersize = 4, clip_on = False, alpha=.5)

# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$t$', size=20)
# ax.legend(loc='upper center', bbox_to_anchor=(0.9, -0.05), ncol=5, frameon=False, prop={'size': 15})
# ax.set_title('$u(x,t)$', fontsize = 20)
# ax.tick_params(labelsize=15)

# plt.show()