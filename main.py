# referencing my old network training set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from numpy import genfromtxt
from torch.utils.data import DataLoader,TensorDataset
from training_function import training_loop, test_score

# Declare a network
class net(nn.Module):

    def __init__(self):
        super().__init__()

        self.classi = nn.Sequential(
            nn.Linear(7,256),
            nn.Linear(256,256),
            nn.Linear(256,256),
            nn.Linear(256,256),
            nn.Linear(256,128),
            nn.Linear(128,64),
            nn.Linear(64,32),
            nn.Linear(32,2)
        )
    def forward(self,inputs):
        output = self.classi(inputs)

        return output

network = net()
network.to('cuda')
print(net)

# load data
folder = "75_dataset"
train_input = torch.load(f"data/{folder}/data_train_input.csv",weights_only=False)
train_label = torch.load(f"data/{folder}/data_train_label.csv",weights_only=False)
test_input = torch.load(f"data/{folder}/data_test_input.csv",weights_only=False)
test_label = torch.load(f"data/{folder}/data_test_label.csv",weights_only=False)
print(train_input.shape)
print(train_label.shape)
print(test_input.shape)
print(test_label.shape)

# Create dataloader
dataset = TensorDataset(train_input,train_label)
train_data = DataLoader(
    dataset = dataset,
    batch_size = 24,
    num_workers = 0,
    shuffle = True,
    pin_memory = True
)

w1 = 0.18
w2 = 1.8
step = 0.01
optimizer = optim.SGD(network.parameters(),lr=0.0001)

print("Start trainging")
for epoch in range(2000):
    print(f"epoch: {epoch} ,w1: {w1:.5f} ,w2: {w2:.5f}")
    network,average_loss = training_loop(network=network,
                                       dataloader=train_data,
                                       optimizer=optimizer,
                                       weight1=w1,
                                       weight2=w2
                                       )
    print(f"Loss: {average_loss:.3f}")
    acc_good,acc_bad = test_score(network=network,test_set=test_input)

    w1 -= (acc_good-acc_bad)*step/100
    w2 += (acc_good-acc_bad)*step/100
    if w1 < 0 : w1 = 0
    if w2 < 0 : w2 = 0

print("Finished training")

# Save the network
PATH = './cifar_net.pth'
torch.save(network.state_dict(), PATH)