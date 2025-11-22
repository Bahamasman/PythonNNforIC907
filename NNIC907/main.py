import numpy as np
import matplotlib.pyplot as plt
import time
import myNN as nn

# Generate your existing data
xpts = np.linspace(0,1,50)
npts = len(xpts)
yreal = np.sin(30*xpts) + np.cos(10*xpts) + 2 + xpts**2

# Prepare data
X = xpts.reshape(npts,1)
y = yreal.reshape(npts,1)

# Create network
NeuralNetwork = nn.myNeuralNetwork()
hidden_size = [100,100]
num_hiddenLayers = len(hidden_size)
input_size = 1
output_size = 1
rate = 1.0 
activationf = 'tanh'
#Options for activationf: 'ReLU', 'LeakyReLU', 'tanh', 'sigmoid'

# Training parameters
epochs = 10001
lr = 0.01

NeuralNetwork.build(input_size,output_size,num_hiddenLayers,hidden_size,activationf,rate)
NeuralNetwork.initialize()
NeuralNetwork.defineLossFunction()
NeuralNetwork.defineOptimizer('Adam', lr)
#Options for optimizer: 'SGD', 'SGD_Decay', 'AdaGrad', 'Adam'

# Splitting data
X_train, X_test, y_train, y_test = NeuralNetwork.train_test_split(
    X, y, test_size=0.2, random_state=42
)


train_losses = []
test_losses = []
print("\nStarting training...")

# Start timing
start_time = time.time()

# Training loop
for epoch in range(epochs):
    # Forward pass
    train_pred = NeuralNetwork.forward(X_train)

    # Compute loss (mean squared error)
    train_loss = NeuralNetwork.computeLoss(y_train)

    # Backward pass
    NeuralNetwork.backward(y_train)

    # Update parameters
    NeuralNetwork.updateParam()
    
    # Store loss and parameters for plotting
    train_losses.append(train_loss)
    
    # Evaluate on TEST data
    test_pred = NeuralNetwork.forward(X_test)
    test_loss = NeuralNetwork.computeLoss(y_test)
    test_losses.append(test_loss)
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

# End timing
end_time = time.time()
training_time = end_time - start_time
print("\nTraining Ended.")

# Final evaluation
print(f"\n=== Final Results ===")
print(f"Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
print(f"Final Train Loss: {train_losses[-1]:.6f}")
print(f"Final Test Loss: {test_losses[-1]:.6f}")

# Calculate ratio evolution for all epochs first
train_test_ratios = []
for i in range(len(train_losses)):
    if train_losses[i] > 0:  # Avoid division by zero
        ratio_i = test_losses[i] / train_losses[i]
        train_test_ratios.append(ratio_i)
    else:
        train_test_ratios.append(1.0)

# Check overfitting
ratio = test_losses[-1] / train_losses[-1]
print(f"Test/Train Ratio: {ratio:.3f}")

# Additional ratio analysis
min_ratio = min(train_test_ratios)
max_ratio = max(train_test_ratios)
avg_ratio = np.mean(train_test_ratios)

print(f"\nRatio Analysis:")
print(f"Minimum ratio: {min_ratio:.3f}")
print(f"Maximum ratio: {max_ratio:.3f}")
print(f"Average ratio: {avg_ratio:.3f}")
print(f"Ratio stability: {max_ratio - min_ratio:.3f} (lower is more stable)")

# Use the evaluate method for comprehensive metrics
test_results = NeuralNetwork.evaluate(X_test, y_test)
print(f"Test R² Score: {test_results['r2_score']:.6f}")
print(f"Test MAE: {test_results['mae']:.6f}")

# Visualization with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Left: Show data split
axs[0].scatter(X_train.flatten(), y_train.flatten(), alpha=0.6, label='Training Data', s=30)
axs[0].scatter(X_test.flatten(), y_test.flatten(), alpha=0.8, label='Test Data', s=30)
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Data Split Visualization')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Middle: Training vs Test Loss
axs[1].plot(train_losses, label='Training Loss', alpha=0.8)
axs[1].plot(test_losses, label='Test Loss', alpha=0.8)
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Loss')
axs[1].set_title(f'Training vs Test Loss')
axs[1].set_yscale('log')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Add comprehensive metrics text box in the performance plot
metrics_text = f'Test/Train Ratio: {ratio:.3f}'
axs[1].text(0.02, 0.98, metrics_text, transform=axs[1].transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
             verticalalignment='top', fontsize=10, fontweight='bold')

# Right: Final predictions on test set
x_plot = np.linspace(0, 1, 200)
X_plot = x_plot.reshape(-1, 1)
y_plot_pred = NeuralNetwork.forward(X_plot)

axs[2].plot(x_plot, np.sin(30*x_plot) + np.cos(10*x_plot) + 2 + x_plot**2, 
           'g-', label='True Function', linewidth=2)
axs[2].plot(x_plot, y_plot_pred.flatten(), 'r-', label='NN Prediction', linewidth=2)
axs[2].scatter(X_test.flatten(), y_test.flatten(), 
              alpha=0.7, color='orange', label='Test Data', s=40, edgecolor='black')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].set_title('Model Performance on Test Data')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Add comprehensive metrics text box in the performance plot
metrics_text = f'R² Score: {test_results["r2_score"]:.4f}\nMAE: {test_results["mae"]:.4f}\nTraining Time: {training_time:.1f}s'
axs[2].text(0.02, 0.98, metrics_text, transform=axs[2].transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
             verticalalignment='top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
