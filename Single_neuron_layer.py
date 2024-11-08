#!/usr/bin/env python
# coding: utf-8

# <H1>Single Neuron code </h1>
# 

# In[2]:


input1 = 3.4
weight1 = 0.2
bias1 = 0.1

neuron1 = (input1*weight1) + bias1
print(neuron1)


# <h1> layer of neuron</h1>
# it might sound fancy but if we dig down it simple dot product of input matrix and weight of each input for that neuron, let's look into code

# In[29]:


input_mat = [[1,2,3,4],
            [2,3,4,5],
            [5,6,7,8],
            [8,9,3,4]]
weight_mat = [[0.1,0.2,0.2,-0,2],
             [-0.1,0.03,0.02,0.04],
             [0.4,0.5,0.6,0.7],
             [0.01,0.02,0.03,0.04]]

bias = [0.001,0.234,0.45,-0.67]
#neuronL2 = []


# In[35]:


neuronL1 = []
for inp1,weigh1,bias_n1 in zip(input_mat,weight_mat,bias):
    #neuronL1 = []
    temp_Res = 0
    for input_n1,weight_w1 in zip(inp1,weigh1):
        temp_Res = temp_Res + (input_n1*weight_w1)
    #print(bias_n1)
    print(temp_Res)
    neuronL1.append(temp_Res+bias_n1)
print(neuronL1)


# In[ ]:




