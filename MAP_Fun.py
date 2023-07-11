import numpy as np
from numpy.linalg import norm

def F(X,Xn,T):
    return np.log(  norm(np.dot(T, X ) - Xn) ** 2  )
 
def F_der(X,Xn,T):
    return 2 * (np.dot(T.T , (np.dot(T,X) - Xn) ) ) / ( norm(np.dot(T, X ) - Xn) ** 2 + 1e-8)