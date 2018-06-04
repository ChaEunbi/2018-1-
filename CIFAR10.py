
# coding: utf-8

# In[4]:

import torch
import torchvision
import torchvision.transforms as transforms


# In[5]:

batchsize=8
epoch_num=5
learning_rate=0.0002


# In[6]:

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='~/eunbi', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='~/eunbi', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[7]:

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

net = Net()

if torch.cuda.is_available():
    net.to(device)


# In[ ]:

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# In[ ]:

#Training the network

if torch.cuda.is_available():
    for epoch in range(epoch_num):  
        for i,(images, labels) in enumerate(trainloader):
            images=images.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()

            #Forward Backward Optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if i%1000==0:
                print('Number of epochs: %d, Mini Batch order: %d' %(epoch+1,i))
                #torch.save(model,'./cifar_model.pkl')              
else:
    for epoch in range(epoch_num):  
        for i, data in enumerate(trainloader,0):
            inputs, labels = data 
            
            optimizer.zero_grad()

            #Forward Backward Optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

                # print statistics
            if i%1000==0:
                print('Number of epochs: %d, Per Batch: %d' %(epoch+1,i))
                #torch.save(model,'./cifar_model.pkl')

print('Finished Training')


# In[ ]:

#Test data
with torch.no_grad():
    num_correct = 0
    total_data = 0
    if torch.cuda.is_available():
        for images, labels in testloader:
            images=images.to(device)
            labels=labels.to(device)
            output = net(image)
            _, expected = torch.max(output.data, 1)

            total_data += labels.size(0)
            num_correct += (expected == labels).sum().item()
    else:
        for data in testloader:
            image, label=data
            output = net(image)
            _, expected = torch.max(output.data, 1)

            total_data += label.size(0)
            num_correct += (expected == label).sum().item()
        
print('Accuracy of the Data: %d %%' % (100 * num_correct / total_data))


# In[ ]:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, expected = torch.max(output, 1)
        c = (expected == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:




# In[ ]:




# In[ ]:



