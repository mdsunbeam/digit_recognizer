######################################## IMPORTS #########################################################################

import numpy as np

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

train_df = pd.read_csv("train.csv")
train_labels = train_df.iloc[:, 0]
train_images = train_df.iloc[:, 1:]

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28)), transforms.Normalize(0.5, 0.5)])

train_labels = torch.tensor(train_labels.values)
train_images = transform(train_images)

class digits_dataset(Dataset):
    def __init__(self):
        self.train_labels = train_labels
        self.train_images = train_images

    def __len__(self):
        return len(train_labels)

    def __getitem__(self, idx):
        return (self.train_images[idx], self.train_labels[idx])
############################################ MODEL DEFINTION #############################################################

class cnn_digit_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(28, 14, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(14, 7, 2)
        self.Dropout = nn.Dropout()
        self.fc1 = nn.Linear(7 * 2 * 2, 14, 4)
        self.fc2 = nn.Linear(14, 7, 2)
        self.fc3 = nn.Linear(7, 1, 2)

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

dataset = cnn_digit_net()
train_dataloader = DataLoader(dataset, batch_size=1024)
################################## TRAINING LOOP ############################################################################

t_0 = time.time()
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(train_dataloader):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.float)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000==1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

time_elapsed = time.time() - t_0
print(f'Training Finished. It took {time_elapsed:5d} seconds.')

