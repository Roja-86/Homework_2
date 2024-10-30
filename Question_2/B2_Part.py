import numpy as np
import matplotlib.pyplot as plt

# ReLU function and its derivative
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# Mean squared error function
def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Neural network initialization
np.random.seed(42)

# Define input (X) and target output (Y)
X = np.array([[-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]).T
Y = np.array([[-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321]]).T

# Network architecture
n_inputs = 1
n_hidden1 = 10
n_hidden2 = 10
n_outputs = 2
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
W1 = np.random.randn(n_hidden1, n_inputs)
b1 = np.zeros((n_hidden1, 1))
W2 = np.random.randn(n_hidden2, n_hidden1)
b2 = np.zeros((n_hidden2, 1))
W3 = np.random.randn(n_outputs, n_hidden2)
b3 = np.zeros((n_outputs, 1))

# Store loss history
loss_history = []

# Training loop
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    Z3 = np.dot(W3, A2) + b3
    A3 = Z3  # No activation in output layer (linear activation for regression)
    
    # Compute loss
    loss = mse_loss(Y.T, A3)
    loss_history.append(loss)
    
    # Backpropagation
    dZ3 = A3 - Y.T
    dW3 = np.dot(dZ3, A2.T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    
    # Update weights and biases
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    # Plot approximations at certain epochs
    if epoch in [9, 99, 199, 399, 999]:
        plt.figure()
        plt.plot(X, Y[:, 0], label='True function', color='b')
        plt.plot(X, A3[0, :], label=f'NN Approximation at Epoch {epoch + 1} (output 1)', linestyle='dashed', color='r')
        plt.legend()
        plt.title(f"Function Approximation at Epoch {epoch + 1}")
        plt.show()

# Plot training error vs epoch
plt.plot(range(epochs), loss_history)
plt.xlabel('Epochs')
plt.ylabel('Training Error (MSE)')
plt.title('Training Error vs Epoch Number')
plt.show()
