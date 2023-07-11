#------ 1066600 --------#
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

import ML_Fun as ml
import MAP_Fun as map
import Load_files as lf
#----------------------------------------#
def main():
    Xn,Xi = lf.load_Distorted() # Load Distorted 8s
    N = 500 # How many pixels to keep
    image = 0; # 0 to 3
    Xi = Xi[:,image]
    Xn = np.reshape( Xn[:N,image], (N,1) )
    
    T = np.concatenate( (np.eye(N, N), np.zeros((N, 784-N)) ), axis = 1) # Transormation matrix for MAP
        
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

    # Generator weights
    A1,A2,B1,B2 = lf.load_Gen()
    
    # Trainning Loop
    for iteration in range(num_iterations):
        # Front propagation
        W1 = np.dot(A1,Z) + B1
        Z1 = ml.relu(W1)
        W2 = np.dot(A2, Z1) + B2
        X = ml.sigmoid(W2)
        
        # Calculate Gradients
        U2 = map.F_der(X,Xn,T)
        V2 = np.multiply(U2, ml.sigmoid_der(W2))
        U1 = np.dot(A2.T , V2)
        V1 = np.multiply(U1, ml.relu_der(W1))
        U0 = np.dot(A1.T , V1)
        grad_J = N * U0 + 2 * Z

        # Cost
        Cost_matrix.append( N * map.F(X,Xn,T) + norm(Z)**2 )

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
    
if __name__ == "__main__":
    main()