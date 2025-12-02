from main import *

Layers = [2, 3, 4]
Neurons = [16, 32, 64]
Epochs = [2000, 3000, 4000]
nCollocations = [20000]

Learning_Rate = 0.001
nSamples = 5000
Noise = [0.0, 0.001]

training_times = []
errors_u = []
errors_E = []

iter = 1


X_train, u_train = set_up_training_set(Noise[0])

for layers in Layers:
    for neurons in Neurons:
        for epochs in Epochs:
                for nCollocation in nCollocations:
                    train_time, error_u, error_E = Simulation(layers, neurons, epochs, nCollocation, X_train, u_train, iter=str(iter))
                    training_times.append(train_time)
                    errors_u.append(error_u)
                    errors_E.append(error_E)
                    iter += 1

# Write results to a file
with open("results.txt", "a") as f:
    for time in training_times:
        f.write(f"{time} ")
    f.write("\n")
    for error_u in errors_u:
        f.write(f"{error_u} ")
    f.write("\n")
    for error_E in errors_E:
        f.write(f"{error_E} ")
    f.write("\n")