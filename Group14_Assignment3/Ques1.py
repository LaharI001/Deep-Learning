#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[4]:


import numpy as np
from PIL import Image
import cv2
import random
import os
from IPython import display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[ ]:





# In[5]:


def convolution(matrix, kernal):
    result = []
    n1 = matrix.shape[0] - kernal.shape[0] + 1
    n2 = matrix.shape[1] - kernal.shape[0] + 1
    for i in range(n1):
        temp = []
        for j in range(n2):
            value = 0
            for k in range(kernal.shape[0]):
                for m in range(kernal.shape[1]):
                    value = value + matrix[i+k][j+m]*kernal[k][m]
            temp.append(value)
        result.append(temp)
    result = np.array(result)
    return result


# In[33]:


kernel = []
for i in range(3):
    kernel.append(np.random.normal(0, (2**0.5)/(9**0.5), (3,3)) )
def Q1(images):
    for j in range(3):
        img = Image.open(images).convert('L')
        imageArray = np.array(img)
        imageArray = cv2.resize(imageArray,(224,224))
        imageArray = np.pad(imageArray, 1)
        img = Image.fromarray(imageArray)

        kernal = kernel[i]
        print("Filter ",j+1)
        print(kernal)
        print()
        plt.imshow(kernal, cmap = "gray")
        plt.show()
        convolutatedImage = convolution(imageArray, kernal)
        img3 = Image.fromarray(convolutatedImage)
        implot = plt.imshow(img3, cmap  = "gray")
        plt.show()


# In[34]:


img1 = "./Group_14/train/grand_piano/image_0001.jpg"
img2 = "./Group_14/train/starfish/image_0002.jpg"
img3 = "./Group_14/train/trilobite/image_0001.jpg"


# In[35]:

print("Image 1: Grand Piano")
Q1(img1)


# In[36]:

print("Image 2: StarFish")
Q1(img2)


# In[37]:

print("Image 3: Trilobite")
Q1(img3)


# In[ ]:




