import numpy as np

# custom loss to penalize negative residuals (contributions to HH) twice as much
def assym_loss(y_true, y_pred):
    x = y_true - y_pred
    grad = -2 * x / np.where(x < 0, 0.5, 1)
    hess = 2 / np.where(x < 0, 0.5, 1)
    return grad, hess