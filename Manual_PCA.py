import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

def PCA_Manual():
    A = pd.read_csv("Complete_Data_.csv")
    print(A)
    M = mean(A.T, axis=1)
    print("\nMean\n",M)
    C = A - M
    V = cov(C.T)
    print("Covariance\n",V)

    values, vectors = eig(V)
    print("Eigen Vectors\n",vectors)
    print("Eigen Values\n",values)

    P = vectors.T.dot(C.T)
    print(P.T)

PCA_Manual()
