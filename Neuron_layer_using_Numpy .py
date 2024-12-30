#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[21]:


input1 = [[1,2.3,3,4.6],
        [2.3,4.5,3.4,5.6],
        [4.5,8.9,13,5.6],
        [3.5,77,6.6,5.5]]

#input matrix


# In[22]:


weight1 = [[2.2,2.1,2.3,2.4],
         [3.4,3.3,3.6,3.7],
         [5.6,5.5,5.6,5.7],
         [2.3,2.5,2.6,2.7]]


# In[23]:


bias1 = [2,2,4,2]


# In[24]:


inputnp = np.array(input1)
weightnp = np.array(weight1)
biasnp = np.array(bias1)


# In[33]:


output1 = np.dot(inputnp,weightnp.T) +biasnp
print(output1)


# In[18]:


type(inputnp)


# In[19]:


type(weightnp)


# <h1> lets try multiple layer of neurons</h1>

# In[28]:


weight2 = [[3.2,2.1,4.3,2.4],
         [3.4,3.3,3.6,3.7],
         [5.6,7.5,7.6,2.7],
         [9.3,2.5,3.6,1.7]]


# In[29]:


Bias2 = [4,5,6,7]


# In[30]:


weightnp2 = np.array(weight2)
biasnp2 = np.array(Bias2)


# In[34]:


first_layer = np.dot(inputnp,weightnp.T) + biasnp
    
    


# In[35]:


second_layer = np.dot(first_layer,weightnp2.T) + biasnp2


# In[36]:


print(second_layer)


# In[ ]:




