from pinn import *
#TODO Check the pde
#TODO Pass NN loss as optional
#TODO Add BCs
#TODO Build more cases
#TODO Make better way to inform A, L, rho, E, f maybe a .json?
#TODO Choose one way to build NN

############################################# Initial Configurations #######################################################
input_path = "/home/marina/programming/NeuralNetwork-Research/PythonNNforIC907/PINN/"
input_file = "results.json"
input_data = input_path+input_file #! DO THIS IN A SMATER WAY

with open(input_data) as myFile:
    mathematica_data = json.load(myFile)

# physics informed neural network f(x, t)
L = to_float(mathematica_data["Properties"]["L"])
Interval = to_float(mathematica_data["Properties"]["Interval"])
A = to_float(mathematica_data["Properties"]["A"])
E_type = mathematica_data["Properties"]["E"]
rho_type = mathematica_data["Properties"]["rho"]
f_type = mathematica_data["Properties"]["f"]

def E(x:torch.Tensor):
  if E_type == "Constant":
    return 2.0
  elif E_type == "Polynomial":
    return x
  elif E_type == "Piecewise":
    return torch.where(x < 2., 2.0, 5.0) 

def rho(x:torch.Tensor):
  if rho_type == "Constant":
    return 1.0
  elif rho_type == "Polynomial":
    return x
  elif rho_type == "Piecewise":
    return torch.where(x < 2., x, 30.0) # allows for element-wise selection from two tensors based on a boolean condition

def f(x:torch.Tensor):
  if f_type == "Constant":
    return 2.0
  elif f_type == "Polynomial":
    return 2*x
  elif f_type == "Piecewise":
    return torch.where(x < 2., x**2, 5.0)

# u_x0 = 
# u_t0 = 

def net_physics(model:torch.nn.Module, x, t): #! ADD BC

  elas = model.elas
  # rho = model.rho
  u = model.net_u(x, t)

  dudt = grad(u, t)
  d2udt2 = grad(dudt, t)
  dudx = grad(u, x)
  d2udx2 = grad(elas*dudx, x) #! CHECK

  pde = rho(x) * A * d2udt2 - A * d2udx2 - f(x)

  return pde


def net_bc(model:torch.nn.Module): 
  x_0 = np_to_th(np.array([0.0])).requires_grad_(True) # x=0
  t_0 = np_to_th(np.array([0.0])).requires_grad_(True) # t=0
  u_0_pred = model.net_u(x_0, t_0)

  return u_0_pred


# For reproducibility
np.random.seed(1234)

NN_infos = [2, 20, 1, 4, 3000, 0.01] # input_size, hidden_size, output_size, depth, epochs, learning_rate
layers = [2, 20, 20, 20, 20, 1]

# Exact solution data set
x_domain = np.linspace(0., 4., 1000)
t_domain = np.linspace(0., 60., 1000)
x = np.array(to_float(mathematica_data["x"])).flatten()[:,None]
t = np.array(to_float(mathematica_data["t"])).flatten()[:,None]
u = np.array(to_float(mathematica_data["u"])).T

X, T = np.meshgrid(x,t)

# Given two one-dimensional arrays, x and y, np.meshgrid(x, y) returns two 2-D arrays, X and Y.
# The array X contains the x-coordinates of all points on the grid, with x values repeated row-wise.
# The array Y contains the y-coordinates of all points on the grid, with y values repeated column-wise.

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # empilha arrays horizontalmente (em colunas)
u_star = u.flatten()[:,None]
UI = griddata(X_star, u_star, (X, T), method='cubic') #! APAGAR

# Domain bounds
Xmin = X_star.min(0)
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
noise = 1.0
sample = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
X_train = X_star[sample,:]
u_train = u_star[sample,:] + np.random.uniform(-noise, noise, nSamples).T
# u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

# Training
model = PINN_DynamicBar(X_train, u_train, NN_infos, Xmin, Xmax, layers, pde=net_physics)
losses = model.train()

# Evaluation
u_pred = model.predict(X_star)
U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
# error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

elas_value = model.elas.detach().cpu().numpy() # detach takes the grad tracking and item gives the raw value (not a tensor but a float)
#error_E = np.abs(elas_value - E(np_to_th(x))) * 100

plot_loss(losses)
#print('Error E: %.5f%%' % (error_E))
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