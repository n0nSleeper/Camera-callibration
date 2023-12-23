import numpy as np 
import cv2 as cv
import pandas as pd 
import os
from utils import *

ob_color = [224,185,85] #object color

path = "cropped\\"
dest1 = "processed_pca\\"
dest2 = "processed_moments\\"

img_list = os.listdir(path)
samples_size = len(img_list)
img_names = np.arange(1, samples_size+1)
X_pca = []
X_moments = []
for img_name in img_names:
    rpath = path + str(img_name) + ".jpg"
    img = cv.imread(rpath)
    #print(rpath)

    #create image mask using hsv color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lb, ub = get_limits(ob_color)
    mask = cv.inRange(hsv, lb, ub)
    mask = cv.erode(mask, None, iterations= 2)
    mask = cv.dilate(mask, None, iterations= 10)
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    #compute center in 2 ways and store them in 2 seperate lists
    pca_center = center_pca(mask)
    moment_center = center_moments(mask)
    X_pca.append(pca_center)
    X_moments.append(moment_center)

    # checking if center match the center of the object
    radius = 10
    ccolor = (0, 255, 0)
    c1 = cv.circle(img, pca_center, radius, ccolor, -1)
    c2 = cv.circle(img, moment_center, radius, ccolor, -1)
    name1 = dest1 + str(pca_center[0]) + "_" + str(pca_center[1]) + ".jpg"
    name2 = dest2 + str(moment_center[0]) + "_" + str(moment_center[1]) + ".jpg"
    cv.imwrite(name1, c1)
    cv.imwrite(name2, c2)

X_pca = np.array(X_pca)
X_moments = np.array(X_moments)
# world coordinate
X = []
Y = []
with open('coord.txt', 'r')  as f:
    lines = f.readlines()
    for line in lines: 
        line = str(line)
        line = line.replace("\n", "")
        ls = line.split(" ")
        X.append(int(ls[0]))
        Y.append(int(ls[1]))

X = np.array(X, dtype="float")*1.05
Y = np.array(Y, dtype="float")*1.05

#save everything to a pandas dataframe 
df = pd.DataFrame()
df["u_pca"] = X_pca[:, 0]
df["v_pca"] = X_pca[:, 1]
df["u_moments"] = X_moments[:, 0]
df["v_moments"] = X_moments[:, 1]
df["x_world"] = X
df["y_world"] = Y
df.to_csv("data.csv", index=False)



