import torch
import torchvision
import tqdm
import time
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
print("Multiple GPU test")
print("There is", torch.cuda.device_count(), "GPUs avaiable!")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dev_name0 = torch.cuda.get_device_name(device=0)
dev_name1 = torch.cuda.get_device_name(device=1)
print("Device 0 {}".format(dev_name0))
print("Device 1 {}".format(dev_name1))
print(device)

device0 = torch.device('cuda:0')
device1 = torch.device('cuda:1')
print(device0)
print(device1)

#print(torch.cuda.get_device_name(device))


batchsize = 2
split = 2 #devider and not the number of mini batch

print("Use of {} GPU and batch size is {}".format(torch.cuda.device_count(),batchsize))
print("split size is {}".format(split))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

start_time = time.time()
print("Start timer")

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = torch.nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool
        ).to(device0)

        self.seq2 = nn.Sequential(
            self.conv2,
            self.relu,
            self.pool
        ).to(device1)


        self.fc = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3
        ).to(device1)

    def forward(self, x):
        x = self.seq2(self.seq1(x).to(device1))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc(x)
        return x

class PipelineParallelNet(Net): # inherit from the existing Net module
    def __init__(self, split_size=split, *args, **kwargs):
        super(PipelineParallelNet, self).__init__(*args, **kwargs)
        self.split_size = split_size
        print(split_size)

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits) #take the first mini batch
        s_prev = self.seq1(s_next) #go through the sequ1 with the first mini batch and the result goes in s_prev assigned with device 1
        # print("s_prev batch 0 device out of seq1 {}".format(s_prev.device))
        s_prev = s_prev .to(device1)
        # print("s_prev device is {}".format(s_prev.device))
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1

            s_prev = self.seq2(s_prev) #the first mini batch continue through seq 1 with device 1
            # print("s_prev device out of seq2 {}".format(s_prev.device))
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))


            # B. s_next runs on cuda:0, which can run concurrently with A)
            s_prev = self.seq1(s_next)
            # print("s_prev batch 1 device out of seq1 {}".format(s_prev.device))
            s_prev = s_prev.to(device1)



        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)

# net = Net()
net = PipelineParallelNet()
# net = net.to(device)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0

    # loop = tqdm.tqdm(trainloader,0)

    for i, data in enumerate(trainloader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device0))
        # print("Output device is {}".format(outputs.device))
        labels = labels.to(outputs.device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
print("--- %s seconds ---" % round(time.time() - start_time))
