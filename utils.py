import numpy as np
import cv2 as cv


def get_limits(color):
    c = np.uint8([[color]])  
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    if hue >= 165:  
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def down_scale(image, dw, dh):
    dim = (image.shape[1], image.shape[0])
    reshaped_dim = (int(dim[0]*dw), int(dim[1]*dh))
    reshaped = cv.resize(image, reshaped_dim, interpolation=cv.INTER_AREA)
    return reshaped

def center_pca(mask):
    mat = np.argwhere(mask != 0)
    mat[:, [0, 1]] = mat[:, [1, 0]]
    mat = np.array(mat).astype(np.float32) 
    m, _ = cv.PCACompute(mat, mean = np.array([]))
    center = m[0]
    center = np.array(center).astype(np.int32)
    return center


def center_moments(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    M = cv.moments(contours[0])
    center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return center