import numpy as np

# Father class (for any type of loss)
class Loss:

  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y):

    # Calculate sample losses
    sample_losses = self.forward(output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Return loss
    return data_loss


# MSE loss class
class Loss_MeanSquaredError(Loss):
  def forward(self, y_pred, y_true):

    # Calculate loss
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

    return sample_losses

  def backward(self, dvalues, y_true): #note that dvalues = y_pred
    nsamples = len(dvalues)
    noutputs = len(dvalues[0])

    # Recall
    # Loss = 1/nsamples * sum(L_i)
    # L_i = 1/noutputs * sum((y_true_i - y_pred_i)**2)
    self.dinputs =  -2 * (y_true - dvalues) / noutputs # Gradient of outputs
    self.dinputs = self.dinputs / nsamples  # normalized by the samples