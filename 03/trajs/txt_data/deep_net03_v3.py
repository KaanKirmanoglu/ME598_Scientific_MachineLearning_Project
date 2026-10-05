import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time

"""
What is the best way to load the data to gpu to do the training?
Not so straightforward with minibatche+dataloader

Should we use minibatch?
How many batches?

v3 : Try with Gpu, transfer data to gpu at each batch per each epoch not before
What to do with Pin memory :
https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

If your data is small (like ours at the moment) you can directly put all the data
to GPU, then there is no point in setting pin_memory to true
---
If you load your data from cpu to gpu while training it helps tp set it to True
like this script
"""



class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt("./dataset_kirk03.txt",dtype=np.float32)
        self.n_samples = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])
        self.y_data = torch.from_numpy(data[:,-1])
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))

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
batch_sizes = [128,256,512,1024]
num_workers = [0,1,2,4,6]
pin_memories = [True,False]

for batch_size in batch_sizes:
    for num_worker in num_workers:
        for pin_memory in pin_memories:
             my_dataset = MyDataset()
             loader = DataLoader(dataset = my_dataset, batch_size = batch_size, shuffle = True, num_workers = num_worker,pin_memory=pin_memory)

             data_iter = iter(loader)
             dX,dY = data_iter.next()
             X_shape = dX.size()
             input_size = X_shape[-1]
             model = TheNet(input_size,50,30)
             model.to(device)
             optimizer = torch.optim.Adam(model.parameters(),lr=0.001 )
             criterion = nn.MSELoss()

             N_epochs = 30
             N_iters = 0

             t0 = time.time()
             for i in range(N_epochs):
                 for j,(X_batch,Y_batch) in enumerate(loader):
                     X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                     y_hat = model.forward(X_batch)
                     loss = criterion(y_hat,Y_batch)
                     optimizer.zero_grad()
                     loss.backward()
                     optimizer.step()
                     N_iters += 1
                 if i % 10 == 0:
                     print(f'epoch {i}, loss = {loss}')

             t1 = time.time() - t0
             print("%d epochs in %.2f seconds -script - v3 BatchSize=%d - NumWorker=%d - pin memory=%d" %(N_epochs,t1,batch_size,num_worker,pin_memory))

























exit()
