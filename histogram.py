import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 


image = cv.imread("object_color.jpg")

m_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
blue_color = np.bincount(m_image[:,:, 2].flatten(),minlength=256)
red_color = np.bincount(m_image[:,:, 0].flatten(),minlength=256)
green_color = np.bincount(m_image[:,:,1].flatten(),minlength=256)

plt.imshow(m_image)
plt.show()

plt.subplot(3, 1, 1) 
plt.title("histogram of Blue") 
plt.plot(blue_color, color="blue") 
  
plt.subplot(3, 1, 2) 
plt.title("histogram of Green") 
plt.plot(green_color, color="green") 
  
plt.subplot(3, 1, 3) 
plt.title("histogram of Red") 
plt.plot(red_color, color="red") 
plt.tight_layout() 

plt.show()