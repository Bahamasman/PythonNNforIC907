from pinn import *
import time

#TODO Pass NN loss as optional
#TODO Add BCs
#TODO Normalize E

############################################# Impoting Data #######################################################
input_path = "C:\\Users\\theyd\\OneDrive\\Desktop\\marina\\PythonNNforIC907\\PINN\\InputData\\"
input_file = "data1.json"
input_data = input_path + input_file #! DO THIS IN A SMATER WAY

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
if("u_xL") in mathematica_data["BCs_ICs"]: u_xL = to_float(mathematica_data["BCs_ICs"]["u_xL"])
if("u_t0") in mathematica_data["BCs_ICs"]: u_t0 = to_float(mathematica_data["BCs_ICs"]["u_t0"])
if("du_dx0") in mathematica_data["BCs_ICs"]: du_dx0 = to_float(mathematica_data["BCs_ICs"]["du_dx0"])

def E(x:torch.Tensor):
  if is_numeric(E_type):
    return to_float(E_type)*torch.ones_like(x)
  elif E_type == "Polynomial":
    return x
  elif E_type == "Piecewise":
    return torch.where(x < 2., 2.0, 5.0) 

def f(x:torch.Tensor):
  if is_numeric(f_type):
    return to_float(f_type)*torch.ones_like(x)
  elif f_type == "Polynomial":
    return 2*x
  elif f_type == "Piecewise":
    return torch.where(x < 2., torch.tensor(10000000.0), torch.tensor(0.0)) # allows for element-wise selection from two tensors based on a boolean condition


############################################# Initial Configurations #######################################################
# Dynamic-Bar PDE
def net_physics(model:torch.nn.Module, x, t):

  Elas = model.elas
  # rho = model.rho
  u = model.net_u(x, t)

  dudt = grad(u, t)
  d2udt2 = grad(dudt, t)
  dudx = grad(u, x)
  d2udx2 = grad(Elas*dudx, x) 

  pde = rho * A * d2udt2 - A * d2udx2 - f(x) #! CHECK 
  return pde

#Boundary/Initial Conditions
def net_bc(model):
  n_bc = 200 
  t0 = torch.linspace(0, Interval, n_bc).view(-1, 1).requires_grad_(True) # generate points t to impose BC u_0 = 0
  x0 = torch.zeros_like(t0).requires_grad_(True)
  u_x0_pred = model.net_u(x0, t0)

  xL = L * torch.ones_like(t0).requires_grad_(True)
  u_xL_pred = model.net_u(xL, t0)

  x_init = torch.linspace(0, L, n_bc).view(-1, 1).requires_grad_(True)
  t_init = torch.zeros_like(x_init)
  u_t0_pred = model.net_u(x_init, t_init)

  bc_res = u_x0_pred - u_x0
  # bc_res = u_xL_pred - u_xL
  # bc_res = u_t0_pred - u_t0

  return bc_res

# Exact solution data set
x_domain = np.linspace(0., L, 1000)
t_domain = np.linspace(0., Interval, 1000)
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
Xmin = X_star.min(0) #! COLOCAR PONTOS DA BOUNDARY NOS INPUTS
Xmax = X_star.max(0)

plot_solution(X_star, u_star)


############################################# Training on Non-noisy Data #######################################################
# # Training data set
# nSamples = 5000
# sample = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
# X_train = X_star[sample,:]
# u_train = u_star[sample,:]

# # Training
# model = PINN_DynamicBar(X_train, u_train, NN_infos, Xmin, Xmax, layers, pde=net_physics)
# losses = model.train()

# # Evaluation
# u_pred = model.predict(X_star)
# U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

# elas_value = model.elas.detach().cpu().numpy() # detach takes the grad tracking and item gives the raw value (not a tensor but a float)
# error_E = np.abs(elas_value - E) * 100

# plot_loss(losses)
# print('Error E: %.5f%%' % (error_E))
# plot_predictions(model, X_star, X_train, u_star)


############################################# Training on Noisy Data #######################################################
# Training data set
nSamples = 5000
nCollocations = 5000
noise = 0.0
sample = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
X_train = X_star[sample,:]
u_train = u_star[sample,:] + noise * np.random.randn(nSamples, 1)
# u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

# Neural Network Architecture
depth = 4 # number of layers
width = 30 # number of neurons in the layer
lr = 1e-2
epochs = 3000
NN_infos = [2, width, 1, depth, 1000, lr] # input_size, hidden_size, output_size, depth, epochs, learning_rate
#layers = [2, 30, 30, 30, 30, 1]

print(f"Neural Network Info: \n\t Number of Neurons: {width} \n\t Number of Layers: {depth} \n\t Epochs: {epochs} \n\t Learning Rate: {lr}")

# Training
model = PINN_DynamicBar(X_train, u_train, NN_infos, Xmin, Xmax, pde=net_physics, bc=net_bc, weight_bc=0.2)
initial_time = time.time()
losses = model.train(nCollocations)
end_time = time.time()
total_time = end_time - initial_time
print(f"Training Time: {total_time} s")

# Prediction
u_pred = model.predict(X_star)
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

elas_value = model.elas.detach().cpu().numpy() # detach takes the grad tracking and item gives the raw value (not a tensor but a float)
#error_E = np.abs(elas_value - E(np_to_th(x))) * 100

plot_loss(losses)
#print('Error E: %.5f%%' % (error_E))
print(f"Error u = {error_u}")
plot_predictions(model, X_star, X_train, u_star)


############################################# Plotting Results #######################################################
fig = plt.figure(figsize=(9, 10)) # Creates a new empty figure (the canvas)
ax = fig.add_subplot(111) # Adds a single subplot (an Axes object) to the figure, 1 row, 1 column, 1 subplot

# Displays a 2D array (U_pred.T) as a colored image
h = ax.imshow(U_pred, interpolation='nearest', cmap='rainbow', 
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