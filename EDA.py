#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import cv2
import numpy as np


# In[2]:


data=pd.read_csv('train_labels.csv')


# In[3]:


bla=data.iloc[:,1].values


# In[4]:


import matplotlib.pyplot as plt
positive=(bla==1).sum()
negative=(bla==0).sum()


# In[5]:


plt.bar([1,0], [negative, positive]);
plt.xticks([1,0],["Negative (N={})".format(negative),"Positive (N={})".format(positive)]);


# In[6]:


l1=[]
l2=[]
for i in range(0,bla.size):
    if(bla[i]==0):
        l1.append(data.iloc[i,0])
    if(bla[i]==1):
        l2.append(data.iloc[i,0])


# In[7]:


df=pd.DataFrame({'id':l1[:,:],
                'label' :[0]*89917},index=None)
df2=pd.DataFrame({'id':l2[0:500],
                'label' :[1]*500},index=None)
df=df.append(df2,ignore_index=True)


# In[8]:


Path="E:/Machine Learning/Project/Harit kaggle data/"
df2=pd.DataFrame({'Path':Path+df.iloc[:,0]+'.tif'},index=None)
df=df.join(df2)
df.head()


# In[9]:


import numpy as np


# In[10]:


fig = plt.figure(figsize=(15,7))
for i,idx in enumerate(np.random.randint(0,1000,8)):
    ax = fig.add_subplot(2, 8/2, i+1, xticks=[], yticks=[])
    plt.imshow(plt.imread(df.iloc[idx,2]))
    ax.set_title('Label: ' + str(df.iloc[idx,1]))


# In[11]:


positive = np.zeros([500,96,96,3],dtype=np.uint8)
negative = np.zeros([500,96,96,3],dtype=np.uint8)


# In[12]:


p=0
n=0
for i in range(0,1000):
    if df.iloc[i,1]==1:
        positive[p]=((cv2.imread(df.iloc[i,2])))
        p=p+1
    if df.iloc[i,1]==0:
        negative[n]=((cv2.imread(df.iloc[i,2])))
        n=n+1


# In[13]:


nr_of_bins = 256 #each possible pixel value will get a bin in the following histograms
fig,axs = plt.subplots(3,2,sharey=True,figsize=(10,8))

#RGB channels
axs[0,0].hist(positive[:,:,:,0].flatten(),bins=nr_of_bins)
axs[0,1].hist(negative[:,:,:,0].flatten(),bins=nr_of_bins)
axs[1,0].hist(positive[:,:,:,1].flatten(),bins=nr_of_bins)
axs[1,1].hist(negative[:,:,:,1].flatten(),bins=nr_of_bins)
axs[2,0].hist(positive[:,:,:,2].flatten(),bins=nr_of_bins)
axs[2,1].hist(negative[:,:,:,2].flatten(),bins=nr_of_bins)

axs[0,1].set_ylabel("Red")
axs[1,1].set_ylabel("Green")
axs[2,1].set_ylabel("Blue")

fig.tight_layout()

