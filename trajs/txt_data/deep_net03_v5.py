import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Dataset
import time

"""
Increasing batch size helps to do more epochs per time but it effectively
reduces learning rate so drops the convergence speed 256 is good compromise
for now
---
Why the initial MSE loss is higher with higher batch size?
How then I can compare the loss for different batch sizes?
How should I sum the errors of different batches for the error of each epoch?
(probably take mean)
"""

class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt("./dataset_kirk03.txt",dtype=np.float32)
        self.n_samples = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])
        self.y_data = torch.from_numpy(data[:,-1])
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))

    def load_to_gpu(self,device):
        self.x_data = self.x_data.to(device)
        self.y_data = self.y_data.to(device)

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

class TheNet(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self,input_size, nn_width, num_hidden):
        super().__init__()

        self.layer_first = nn.Linear(input_size, nn_width)
        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)
        self.layer_last = nn.Linear(nn_width, 1)

    def forward(self, x):
        activation = nn.ReLU()
        out = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            out = activation(hidden(out))
        out = self.layer_last(out)
        return out

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

script_name = sys.argv[0]
script_name = script_name[:-3]

batch_size = 256
width = 50
depth = 30
lr = 0.001
N_epochs = 1000

my_dataset = MyDataset()
my_dataset.load_to_gpu(device)
loader = DataLoader(dataset = my_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

data_iter = iter(loader)
dX,dY = data_iter.next()
X_shape = dX.size()
input_size = X_shape[-1]
model = TheNet(input_size,width,depth)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr )
criterion = nn.MSELoss()


t0 = time.time()
losses = []
for i in range(N_epochs):
    epoch_loss = []
    for j,(X_batch,Y_batch) in enumerate(loader):
        y_hat = model.forward(X_batch)
        loss = criterion(y_hat,Y_batch)
        epoch_loss.append(loss.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % 50 == 0:
        with torch.no_grad():
            total_loss = np.array(epoch_loss)
            epoch_loss = np.average(epoch_loss)
        losses.append(epoch_loss)
        print(f'epoch {i}, epoch_loss = {epoch_loss}')

t1 = time.time() - t0
torch.save(model,'./model_%s_lr%.4f_bs%d_Ne%d.pth'%(script_name,lr,batch_size,N_epochs))
print("%d epochs in %.2f seconds -script - v5 BatchSize=%d" %(N_epochs,t1,batch_size))




























exit()
