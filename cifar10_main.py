
# coding: utf-8

# In[2]:


from __future__ import print_function, division


# In[87]:


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision


# In[88]:


plt.ion()


# In[104]:


transform_train = transforms.Compose([
    transforms.RandomCrop(24),
    transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(2, 2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #transforms.RandomAffine(15, shear=15),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.ToTensor(),    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# In[105]:


trainset = torchvision.datasets.CIFAR10(root='.', train=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)


# In[106]:


testset = torchvision.datasets.CIFAR10(root='.', train=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


# In[107]:


num_classes = 10
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[117]:


fig = plt.figure()

for i in range(4):
    image, label = trainset[i]
    
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('#{}: {}'.format(i, classes[label]))
    ax.axis('off')

    image = image / 2 + 0.5     # unnormalize
    npimage = image.numpy()
    npimage = np.transpose(npimage, (1, 2, 0))
    plt.imshow(npimage)

