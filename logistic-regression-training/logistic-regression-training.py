import numpy as np

def sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X,dtype = float)
    y = np.asarray(y,dtype = float)

    # (100,3) - 100 - m and n - 3
    m,n = X.shape

    # weights = [0,0,0,0,...]
    w = np.zeros(n) 
    b = 0.0

    for _ in range(steps):
        # z = wx + b
        z = np.dot(X,w) + b
        y_hat = sigmoid(z)

        # This calculates how much weights should change.
        dw = (1/m)*np.dot(X.T,(y_hat - y))
        # Bias update based on total error.
        db = (1/m)*np.sum(y_hat - y)

        # update parameters
        w -= lr * dw
        b -= lr*db 

    return w,b