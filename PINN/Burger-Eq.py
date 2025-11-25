from pinn import *

# Data-driven Solutions
# Data-driven Discovery

class PINN_Burger():
    def __init__(self, X, u, NN_infos:list, lb, ub, layers=None):
        '''Arguments:
        X(nSamples, input_size): inputs [[xi, ti], ...]
        u(nSamples, output_size): outputs [ui, ...]
        NN_infos = input_size, hidden_size, output_size, depth, epochs, learning_rate
        lb: domain boundary
        ub: boundary
        '''
        # boundary conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        # defining learnable parameters
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # deep neural networks
        self.model = NN(NN_infos[0], NN_infos[1], NN_infos[2], NN_infos[3], layers)
        self.model.register_parameter('lambda_1', self.lambda_1) # explicitly register a torch.nn.Parameter with an nn.Module. This registration ensures that the parameter is included in the module's parameters()
        self.model.register_parameter('lambda_2', self.lambda_2)

        # optimizers
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
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )

    def net_u(self, x, t):
        u = self.model(torch.cat([x, t], dim=1)) # concatenate into model inputs
        return u

    # physics informed neural network f(x, t)
    def net_f(self, x, t):

        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.net_u(x, t)

        dudt = grad(u, t)
        dudx = grad(u, x)
        d2udx2 = grad(dudx, x)

        pde = dudt + lambda_1 * u * dudx - lambda_2 * d2udx2
        return pde

    # pde embeded in the NN through the Loss Function
    def loss_func(self):
        u_pred = self.net_u(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)
        self.optimizer.zero_grad()
        loss.backward()

        self.epochs += 1
        if self.epochs % 100 == 0:
            print(
                'Loss: %e, l1: %.5f, l2: %.5f' %
                (
                    loss.item(),
                    self.lambda_1.item(),
                    torch.exp(self.lambda_2.detach()).item()
                )
            )
        return loss

    def train(self):
        self.model.train() # Set model to training mode
        losses = []

        for epoch in range(self.epochs):
            u_pred = self.net_u(self.x, self.t)
            f_pred = self.net_f(self.x, self.t)
            loss = torch.mean((self.u - u_pred) ** 2) + torch.mean(f_pred ** 2)

            # Backward and optimize
            self.optim_Adam.zero_grad()
            loss.backward()
            self.optim_Adam.step()
            losses.append(loss.item())

            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                    (
                        epoch,
                        loss.item(),
                        self.lambda_1.item(),
                        torch.exp(self.lambda_2).item()
                    )
                )

        # Backward and optimize
        self.optimizer.step(self.loss_func) #using torch.optim.LBFGS
        return losses

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.model.eval() # Set model to evaluation mode
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u

NN_infos = [2, 20, 1, 8, 0, 0.1] # input_size, hidden_size, output_size, depth, epochs, learning_rate
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

data = scipy.io.loadmat('/content/burgers_shock.mat')

# Exact solution data set
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
u_exact = np.real(data['usol']).T
nu = 0.01/np.pi

X, T = np.meshgrid(x,t)
# Given two one-dimensional arrays, x and y, np.meshgrid(x, y) returns two 2-D arrays, X and Y.
# The array X contains the x-coordinates of all points on the grid, with x values repeated row-wise.
# The array Y contains the y-coordinates of all points on the grid, with y values repeated column-wise.

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None])) # empilha arrays horizontalmente (em colunas)
u_star = u_exact.flatten()[:,None]

# Domain bounds
lb = X_star.min(0)
ub = X_star.max(0)

noise = 0.0

# Training data set
nSamples = 5000
idx = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
X_train = X_star[idx,:]
u_train = u_star[idx,:]

# training
model = PINN_DynamicBar(X_train, u_train, NN_infos, Xmin, Xmax, layers, pde=net_physics)
losses = model.train()

# evaluation
u_pred = model.predict(X_star)
# error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

elas_value = model.elas.detach().cpu().numpy() # detach takes the grad tracking and item gives the raw value (not a tensor but a float)
error_E = np.abs(elas_value - E) * 100

plot_loss(losses)
plot_predictions(model, X_star, X_train, u_star)
print('Error E: %.5f%%' % (error_E))


noise = 0.01

# Training data set
nSamples = 2000
idx = np.random.choice(X_star.shape[0], nSamples, replace=False) # generating nSamples random samples from the input data set X
X_train = X_star[idx,:]
u_train = u_star[idx,:]
u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

# training
model = PINN_Burger(X_train, u_train, NN_infos, lb, ub)
losses = model.train()

# evaluations
u_pred = model.predict(X_star)

U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()
lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

#print('Error u: %e' % (error_u))
print('Error l1: %.5f%%' % (error_lambda_1_noisy))
print('Error l2: %.5f%%' % (error_lambda_2_noisy))


fig = plt.figure(figsize=(9, 5)) # Creates a new empty figure (the canvas)
ax = fig.add_subplot(111) # Adds a single subplot (an Axes object) to the figure, 1 row, 1 column, 1 subplot

# Displays a 2D array (U_pred.T) as a colored image
h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
              extent=[t.min(), t.max(), x.min(), x.max()],
              origin='lower', aspect='auto')

# Colorbar axes configs
divider = make_axes_locatable(ax) # “attach” new axes next to an existing one (used for colorbars).
cax = divider.append_axes("right", size="5%", pad=0.10) # Creates a new vertical axis to the right side, with spacing pad
cbar = fig.colorbar(h, cax=cax) # Creates a colorbar showing the mapping between colors and values.
cbar.ax.tick_params(labelsize=15)

# Plot training data points
ax.plot(X_train[:,1],
        X_train[:,0],
        'kx', label = 'Data (%d points)' % (u_train.shape[0]),
        markersize = 4,
        clip_on = False,
        alpha=.5)

# Plot vertical white lines at selected times
# line = np.linspace(x.min(), x.max(), 2)[:,None]
# ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
# ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
# ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(loc='upper center',
          bbox_to_anchor=(0.9, -0.05),
          ncol=5,
          frameon=False,
          prop={'size': 15})
ax.set_title('$u(t,x)$', fontsize = 20)
ax.tick_params(labelsize=15)

plt.show()