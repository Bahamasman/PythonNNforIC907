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

- **Name:**: Physics-Informed Neural Network (PINN) examples for a dynamic bar and Burgers-type problems.
- **Purpose:**: Training PINNs to solve / discover PDEs (dynamic bar elasticity problem and Burger-type PDE examples). The repository contains implementations of neural networks, scaling utilities, training drivers, plotting helpers, and example input data.

**Requirements**
- **Python:**: Python 3.8+ recommended.
- **Libraries:**: `torch`, `numpy`, `scipy`, `matplotlib`. Install with `pip install -r requirements.txt` (see below).

**Quick Install**
- **Create venv:**: `python -m venv .venv`
- **Activate (PowerShell):**:
```powershell
.\.venv\Scripts\Activate.ps1
```
- **Install:**: `pip install torch numpy scipy matplotlib`

**Run (examples)**
- **Train & predict with the dynamic-bar PINN (main driver):**: From the repository `PINN` folder run:
```powershell
python .\main.py
```
- **Burger PDE example:**: The file `Burger-Eq.py` contains a separate PINN/Burgers example; run it similarly:
```powershell
python .\Burger-Eq.py
```

**Important Paths & Data**
- **Input data**: The driver `main.py` loads JSON inputs from a path hardcoded (`input_path` and `input_file`). Example files are in `InputData/` (e.g. `data8.json`). Update the path in `main.py` if necessary.
- **Functions E and f**: These functions have to be changed *manually* when they are `Polynomial` or `Piecewise` with the correct function adopted to generate the solution from Mathematica, so as to be compatible with the input data.
- **Plots**: Generated plots are written to folders referenced in `utils.py` (e.g. `Plots/Solutions`, `Plots/Predictions`, `Plots/Losses`). Note: these paths are currently absolute in the scripts; consider changing them to relative paths (`./Plots/...`) if you want outputs inside this repo.

**Repository Structure**
- **`main.py`**: Main training script for the dynamic-bar PINN. Loads JSON input, prepares training samples, constructs `Scales` and `PINN_DynamicBar`, trains and plots results.
- **`Burger-Eq.py`**: Alternative example implementing a PINN for the Burgers equation (data-driven discovery example).
- **`NNClasses.py`**: Core network definitions: `NN` (MLP), `PINN_DynamicBar` (PINN implementation), training/prediction methods.
- **`ScalesClass.py`**: Scaling utilities to nondimensionalize inputs/outputs and helper conversions between physical and scaled units.
- **`utils.py`**: Helper functions for plotting, gradients, and device selection.
- **`InputData/`**: Example input JSON files (problem definitions and reference solutions).
- **`Plots/`**: Target directories for saved figures (subfolders: `Solutions/`, `Predictions/`, `Losses/`, `E_Predictions/`).

**Input JSON format (expected keys)**
- **`Properties`**: includes `L`, `Interval`, `A`, `rho`, `E` (can be numeric, `Polynomial`, or `Piecewise`), and `f` (numeric, `Polynomial`, or `Piecewise`).
- **`BCs_ICs`**: optional keys such as `u_x0`, `u_xL`, `u_t0`, `du_dx0` for boundary/initial conditions.
- **`x`, `t`, `u`**: arrays with the reference solution grid used for training / evaluation.

**Notes & Tips**
- **Hardcoded absolute paths:**: Several files (`utils.py`, `main.py`) use absolute paths referring to `/home/marina/...`. To run locally on Windows, change those lines to relative paths. 
- **GPU support:**: The code automatically uses CUDA if available. Ensure `torch` is installed with CUDA support for GPU runs.
- **Data noise & sampling:**: `main.py` shows how training samples are drawn (random sampling, added noise). You can change `nSamples`, `noise`, `nCollocations`, and network hyperparameters in `main.py`.
- **Training phases:**: Training uses Adam and then L-BFGS refinement inside `PINN_DynamicBar`.

**Suggested next steps / improvements**
- **Add a `requirements.txt`**: Pin versions for `torch`, `numpy`, `scipy`, `matplotlib`.
- **Make paths relative**: Modify the scripts to use repo-relative paths so running is straightforward.
- **Add CLI**: Provide arguments to `main.py` for selecting `input_file`, epochs, or device.
- **Unit tests / examples**: Add a small smoke test that runs a few training iterations to verify environment correctness.

================================================================================
                              USAGE NOTES
================================================================================

ADAM IMPLEMENTATION:
- Run main.py for a basic demonstration
- Run HyperparameterSensability.py for the Adam sensitivity study
- Change architecture in main.py: hidden_size = [neurons_layer1, ...]
- Choose activation: 'ReLU', 'LeakyReLU', 'tanh', 'sigmoid'

PINN:
- Place or select an input JSON file in ./PINN/InputData/
- Edit or pass --input-file to PINN/main.py
- Run from ./PINN: python main.py
- Check ./PINN/Plots/ for saved figures

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
