import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import cv2 as cv
from sklearn.model_selection import train_test_split


x1_max = 27.3
x2_max = 18.9
u1_max = 2052
u2_max = 1431

ob_height = 1.5
df = pd.read_csv("data.csv")
sample_size = len(df["x_world"])

#processing data into respective matrix
U_pca_ = np.array(df[["u_pca", "v_pca"]], dtype="float64")
_U_pca = np.ones(sample_size, dtype="float64")
U_pca = np.empty((sample_size, 3), dtype="float64")
U_pca[:, :2] = U_pca_
U_pca[:, 2] = _U_pca


X_ = np.array(df[["x_world", "y_world"]], dtype="float64")
_X_= ob_height*np.ones(sample_size, dtype="float64")
_X = np.ones(sample_size, dtype="float64")
X = np.empty((sample_size, 4), dtype="float64")
X[:,:2] = X_
X[:, 2] = _X_ 
X[:, 3] = _X 

#normalize
X[:, 0] = X[:,0]/x1_max
X[:, 1] = X[:,1]/x2_max

U_pca[:, 0] = U_pca[:,0]/u1_max
U_pca[:, 1] = U_pca[:,1]/u2_max

#sampling #23423 
n_sample = 4
X_sample, X_eval, U_pca_sample, U_pca_eval = train_test_split(X, U_pca, test_size=(1-n_sample/sample_size), random_state= 4) 

X_sample = X_sample.T
U_pca_sample = U_pca_sample.T
#solve for camera matrix
M_ = np.dot(np.dot(X_sample, U_pca_sample.T), np.linalg.inv(np.dot(U_pca_sample, U_pca_sample.T)))

#save result matrix as npy file
np.save("M_", M_)

