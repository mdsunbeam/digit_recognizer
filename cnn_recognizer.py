######################################## IMPORTS #########################################################################

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import pandas as pd

import time

####################################### DATA PREP ########################################################################

def generate_data(filepath):
    file = pd.read_csv(filepath)

    img_dataset = []
    if 'train' in filepath:
        label_dataset = file.loc[:, 'label'].values.tolist()
    else:
        label_dataset = []

    for i in range(len(file)):
        initial = 1
        if 'test' in filepath:
            initial = 0
        current = file.loc[i]
        img = np.array(current[initial:len(file.loc[i])])
        img = img.reshape(28, 28)
        img = img / 255
        img_dataset.append(img)

    img_dataset = np.array(img_dataset)
    label_dataset = np.array(label_dataset)
    img_dataset = np.expand_dims(img_dataset, -1)
    label_dataset = np.expand_dims(label_dataset, -1)
    return img_dataset, label_dataset

# generates the train data for easily defining a PyTorch dataset class
train_imgs, train_labels = generate_data("train.csv")

# fig = plt.figure(figsize=(10, 10))

# for i in range(16):
#     fig.add_subplot(4, 4, i+1)
#     plt.title(train_labels[i][0])
#     plt.imshow(train_imgs[i], cmap='Greys_r')
# plt.show()

class digits_dataset(Dataset):
    def __init__(self):
        self.train_labels = torch.from_numpy(train_labels)
        self.train_images = torch.from_numpy(train_imgs)
        self.train_images = self.train_images.permute(0,3,1,2)
        print(self.train_images.shape)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return (self.train_images[idx][:], self.train_labels[idx][0])
############################################ MODEL DEFINITION #############################################################

class cnn_digit_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 1)
        self.pool = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(3, 7, 2)
        self.Dropout = nn.Dropout()
        self.fc1 = nn.Linear(7 * 27 * 27, 56)
        self.fc2 = nn.Linear(56, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.Dropout(x)
        x = F.relu(self.fc2(x))
        x = self.Dropout(x)
        x = self.fc3(x)
        return x
######################################### MODEL, OPTIMIZER & CRITERION INSTANTIATION ########################################

net = cnn_digit_net()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
criterion = nn.MSELoss()
################################# INSTANTIATE DATASET AND DEFINE DATALOADER #################################################

dataset = digits_dataset()
train_dataloader = DataLoader(dataset, batch_size=1024)
################################## TRAINING LOOP ############################################################################

t_0 = time.time()
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = criterion(torch.squeeze(outputs), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2==1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
            running_loss = 0.0

time_elapsed = time.time() - t_0
print(f'Training Finished. It took {time_elapsed:5f} seconds.')

PATH = '/home/sunbeam/code/digit_recognizer/models/cnn_net1.pth'
torch.save(net, PATH)

####################################### TEST SET EVALUATION ###################################################################

# model is loaded in
net = torch.load(PATH)
net.eval()

# generates the test data for easily defining a PyTorch dataset class
test_imgs, test_labels = generate_data("test.csv")
class digits_test(Dataset):
    def __init__(self):
        self.test_labels = torch.from_numpy(test_labels)
        self.test_images = torch.from_numpy(test_imgs)
        self.test_images = self.test_images.permute(0,3,1,2)
        print(self.test_images.shape)

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return (self.test_images[idx][:], self.test_labels[idx][0])

test_data = digits_test()
test_dataloader = DataLoader(test_data, batch_size=1024)

correct = 0
total = 0

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        if outputs == labels:
            correct += 1

        total += 1
        

print(correct/total * 100)