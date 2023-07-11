import scipy.io
import numpy as np

def load_Gen():
    # Load data21.mat file
    mat = scipy.io.loadmat(r"C:\Users\Eygenia\Desktop\GAN\data21.mat")
    A1 = np.array(mat['A_1'])
    A2 = np.array(mat['A_2'])
    B1 = np.array(mat['B_1'])
    B2 = np.array(mat['B_2'])
    
    return A1,A2,B1,B2

def load_Distorted():
    # Load data22.mat file
    mat = scipy.io.loadmat(r"C:\Users\Eygenia\Desktop\GAN\data22.mat")
    Xn = np.array(mat['X_n']) # Distorted
    Xi = np.array(mat['X_i']) # Ideal
    
    return Xn,Xi

def load_downRes():
    # Load data22.mat file
    mat = scipy.io.loadmat(r"C:\Users\Eygenia\Desktop\GAN\data23.mat")
    Xn = np.array(mat['X_n']) # Distorted
    Xi = np.array(mat['X_i']) # Ideal
    
    return Xn,Xi