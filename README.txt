================================================================================
                    NEURAL NETWORK IMPLEMENTATIONS PROJECT
================================================================================

This project contains two main implementations:
1. Custom Neural Network with Adam Optimizer (AdamImplementation/)
2. Physics-Informed Neural Networks for PDEs (PINN/)

================================================================================
                           REQUIRED PACKAGES
================================================================================

COMMON PACKAGES:
- numpy: Numerical computing and array operations
- matplotlib: Data visualization and plotting

ADAM IMPLEMENTATION:
- numpy
- matplotlib

PINN IMPLEMENTATION:
- numpy
- matplotlib
- torch (PyTorch): Deep learning framework
- scipy: Scientific computing (interpolation, I/O)
- json: JSON file handling (included in Python standard library)

INSTALLATION:
pip install numpy matplotlib torch scipy

================================================================================
                        ADAM IMPLEMENTATION FOLDER
================================================================================

This folder contains a custom-built neural network implementation from scratch
using only NumPy, with focus on optimizer comparison and hyperparameter tuning.

FILES:

1. main.py
   - Main training script for basic neural network training
   - Generates synthetic data
   - Demonstrates network training with configurable architecture
   - Includes plotting of predictions vs actual values

2. myNN.py
   - Core neural network class (myNeuralNetwork)
   - Implements forward pass, backward propagation, and training loop
   - Supports multiple hidden layers with customizable sizes
   - Integrates activation functions, loss functions, and optimizers
   - Includes train-test split functionality
   - Loss tracking and history management

3. Activation.py
   - Activation function implementations with forward and backward passes
   - Classes included:
     * Activation_ReLU: Rectified Linear Unit
     * Activation_LeakyReLU: Leaky ReLU with configurable alpha
     * Activation_Sigmoid: Sigmoid activation
     * Activation_TanhH: Hyperbolic tangent

4. LayerDense.py
   - Dense (fully connected) layer implementation
   - Handles weight and bias initialization
   - Forward pass: matrix multiplication with activation
   - Backward pass: gradient computation for weights, biases, and inputs
   - Random initialization: weights and biases

5. Loss.py
   - Loss function implementations
   - Mean Squared Error (MSE) loss for regression tasks
   - Forward pass: computes loss value
   - Backward pass: computes gradients for backpropagation

6. Optimizer.py
   - Optimizer implementations for gradient descent variants
   - Classes included:
     * Optimizer_SGD: Stochastic Gradient Descent
     * Optimizer_Momentum: SGD with momentum
     * Optimizer_Adagrad: Adaptive gradient optimizer
     * Optimizer_RMSprop: Root Mean Square Propagation
     * Optimizer_Adam: Adaptive Moment Estimation (Adam)
   - Features: learning rate decay, parameter update rules
   - Adam optimizer includes beta1, beta2, epsilon parameters

7. HyperparameterSensability.py
   - Comprehensive hyperparameter sensitivity analysis for Adam optimizer
   - Tests 16 different configurations varying:
     * Learning rates: 0.0001 to 0.2
     * Network architectures: 10 to 500 neurons, 1 to 5 layers
     * Activation functions: sigmoid, tanh, LeakyReLU
   - Includes overfitting detection with test/train loss ratio
   - Generates 6 individual plots:
     * Learning rate impact analysis
     * Architecture comparison (size and depth)
     * Activation function comparison
     * Overfitting detection bar chart
     * Performance vs efficiency scatter plot
     * Best vs worst configurations loss evolution

8. IllCondtionedQuadraticFuntion.py
   - Tests optimizer performance on ill-conditioned quadratic functions
   - Compares convergence behavior of different optimizers
   - Useful for understanding optimizer robustness

9. InitialNNcode.py
   - Original/prototype neural network implementation
   - Historical reference for development evolution

================================================================================
                            PINN FOLDER
================================================================================

Physics-Informed Neural Networks (PINNs) for solving partial differential 
equations (PDEs), specifically focused on dynamic bar problems and Burger's 
equation.

FILES:

1. main.py
   - Main training script for PINN models
   - Loads problem configuration from JSON files in InputData/
   - Defines physics problem: bar dynamics with E(x), f(x), boundary/initial conditions
   - Physical parameters: Length (L), Area (A), density (rho), Young's modulus (E)
   - Supports various E(x) and f(x) types: constant, polynomial, piecewise
   - Trains network to satisfy PDE, boundary conditions, and initial conditions
   - Saves plots and results

2. NNClasses.py
   - Neural network architecture definitions using PyTorch
   - NN class: Deep neural network with customizable layers and activation
   - PINN class: Physics-Informed Neural Network
     * Implements PDE residual calculation
     * Handles boundary and initial conditions
     * Automatic differentiation for physics loss terms
     * Training loop with physics-informed loss
   - Supports custom layer configurations or automatic architecture

3. utils.py
   - Utility functions for PINN implementation
   - Functions included:
     * np_to_th: NumPy array to PyTorch tensor conversion
     * to_float: Safe conversion to float
     * Plotting functions for solutions and predictions
     * Grid generation and interpolation utilities
   - PyTorch and NumPy compatibility helpers
   - Random seed setting for reproducibility

4. ScalesClass.py
   - Scales class for non-dimensionalization of PDE problem
   - Computes characteristic scales from physical parameters:
     * Length scale (L)
     * Young's modulus scale (E0)
     * Force scale (F0) from maximum of f(x)
   - Methods for scaling and unscaling variables
   - Ensures numerical stability by normalizing problem domain

5. Burger-Eq.py
   - PINN implementation specifically for Burger's equation
   - PINN_Burger class with learnable PDE parameters (lambda_1, lambda_2)
   - Data-driven discovery of PDE coefficients
   - Implements residual for Burger's equation: u_t + u*u_x - lambda*u_xx = 0
   - Automatic differentiation for spatial and temporal derivatives

6. runSimulations.py
   - Batch execution script for multiple PINN simulations
   - Iterates through different JSON configuration files
   - Automates training across various problem setups
   - Consolidates results from multiple cases

7. README.txt
   - Additional documentation specific to PINN folder

8. results.txt
   - Stored results from PINN training runs
   - Loss values, training times, and performance metrics

InputData/ FOLDER:
   - JSON configuration files (data0.json through data11.json)
   - Each file defines a specific problem case:
     * Physical parameters (L, A, rho)
     * Young's modulus E(x) definition
     * External force f(x) definition
     * Boundary conditions (u at x=0, x=L)
     * Initial conditions (u at t=0)

Plots/ FOLDER:
   - Stores generated visualizations:
     * E_predictions/: Young's modulus predictions
     * Losses/: Training loss curves
     * Predictions/: Network output predictions
     * Solutions/: Final solution plots

================================================================================
                              USAGE NOTES
================================================================================

ADAM IMPLEMENTATION:
- Run main.py for basic training demonstration
- Run HyperparameterSensability.py for comprehensive optimizer analysis
- Modify network architecture in main.py: hidden_size = [neurons_layer1, neurons_layer2, ...]
- Choose activation function: 'ReLU', 'LeakyReLU', 'tanh', 'sigmoid'

PINN:
- Prepare JSON file in InputData/ with problem specification
- Update input_path and input_file in main.py
- Run main.py to train PINN for single case
- Run runSimulations.py to batch process multiple cases
- Check Plots/ folder for generated visualizations

================================================================================
                          PROJECT STRUCTURE
================================================================================

PythonNNforIC907/
├── AdamImplementation/        # Custom NN with Adam optimizer
│   ├── main.py               # Basic training script
│   ├── myNN.py               # Neural network class
│   ├── Activation.py         # Activation functions
│   ├── LayerDense.py         # Dense layer implementation
│   ├── Loss.py               # Loss functions
│   ├── Optimizer.py          # SGD, Momentum, Adam, etc.
│   ├── HyperparameterSensability.py  # Adam sensitivity analysis
│   ├── IllCondtionedQuadraticFuntion.py
│   ├── InitialNNcode.py
│   └── results/              # Experimental results
│
├── PINN/                      # Physics-Informed Neural Networks
│   ├── main.py               # Main PINN training
│   ├── NNClasses.py          # PINN architecture
│   ├── utils.py              # Helper functions
│   ├── ScalesClass.py        # Non-dimensionalization
│   ├── Burger-Eq.py          # Burger's equation PINN
│   ├── runSimulations.py     # Batch execution
│   ├── README.txt
│   ├── results.txt
│   ├── InputData/            # JSON problem configurations
│   └── Plots/                # Generated visualizations
│
└── PROJECT_README.txt         # This file

================================================================================
