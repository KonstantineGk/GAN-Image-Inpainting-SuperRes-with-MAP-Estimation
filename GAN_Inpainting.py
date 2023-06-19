#------ 1066600 --------#
import numpy as np
from numpy.linalg import norm
import scipy.io
import matplotlib.pyplot as plt

#----------------------------------------#
# Load data21.mat file
mat = scipy.io.loadmat('data21.mat')
A1 = np.array(mat['A_1'])
A2 = np.array(mat['A_2'])
B1 = np.array(mat['B_1'])
B2 = np.array(mat['B_2'])

# Load data22.mat file
mat = scipy.io.loadmat('data22.mat')
Xn = np.array(mat['X_n']) # Distorted
Xi = np.array(mat['X_i']) # Ideal
#----------------------------------------#
# Prepare Data
N = 500;
image = 0; # 0 to 3
Xi = Xi[:,image]
Xn = np.reshape( Xn[:N,image], (N,1) );
T = np.concatenate( (np.eye(N, N), np.zeros((N, 784-N)) ), axis = 1)

#-------- Define Functions ---------#
# ReLu
def relu(x):
    return (np.maximum(0,x))

# ReLU derivative
def relu_der_(x):
    if x <= 0: y = 0
    elif x > 0: y = 1
    return y
relu_der = np.vectorize(relu_der_)

# Sigmoid
def sigmoid(x):
    return 1 / ( 1 + np.exp(x) )

# Sigmoid derivative
def sigmoid_der(x):
    return - np.exp(x) / ( (1 + np.exp(x)) ** 2 )

def F(X):
    return np.log(  norm(np.dot(T, X ) - Xn) ** 2  )
 
def F_der(X):
    return 2 * (np.dot(T.T , (np.dot(T,X) - Xn) ) ) / ( norm(np.dot(T, X ) - Xn) ** 2 + 1e-8)
    
#---------------- Main -------------------------#
# Create entry Z from Gaussian
Z = np.random.normal(0,1, 10)
Z = np.reshape(Z,(10,1))

# Define the number of training iterations
num_iterations = 1000

# Define learning rate
learning_rate = 0.008

# Cost matrix
Cost_matrix = []

# Adam parameters
lamda = 0.05
c = 0.001

# Main Loop
for iteration in range(num_iterations):
    # Front propagation
    W1 = np.dot(A1,Z) + B1
    Z1 = relu(W1)
    W2 = np.dot(A2, Z1) + B2
    X = sigmoid(W2)
    
    # Calculate Gradients
    U2 = F_der(X)
    V2 = np.multiply(U2, sigmoid_der(W2))
    U1 = np.dot(A2.T , V2)
    V1 = np.multiply(U1, relu_der(W1))
    U0 = np.dot(A1.T , V1)
    grad_J = N * U0 + 2 * Z

    # Cost
    Cost_matrix.append( N * F(X) + norm(Z)**2 )

    # Adam
    if iteration == 0:P_Z = grad_J ** 2
    else:P_Z = (1 - lamda) * P_Z + lamda * ( grad_J ** 2 )
    
    # Update parameters
    Z -= learning_rate * grad_J / np.sqrt(c + P_Z)

    
# Joined Image
Xn = np.concatenate( (Xn, np.ones((784-N,1)) ), axis = 0)
comp_im = np.zeros((28,28*3))
comp_im[:,0:28] = np.reshape(Xi,(28,28)).T
comp_im[:,28:2*28] = np.reshape(Xn,(28,28)).T
comp_im[:,2*28:] = np.reshape(X,(28,28)).T

# Plot Image
plt.subplot(2, 1, 1)
plt.imshow(comp_im,cmap='gray')

# Plot Cost
plt.subplot(2, 1, 2)
plt.plot(Cost_matrix)
plt.show()
