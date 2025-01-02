#!/usr/bin/env python
# coding: utf-8

# <h1> In built class for creating Sprial Data </h1>

# In[3]:


pip install nnfs


# In[7]:


from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes = 3)
plt.scatter(X[:,0],X[:,1])
plt.show()


# In[8]:


import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


# In[12]:


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
X,y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.output[:5])


# In[ ]:




