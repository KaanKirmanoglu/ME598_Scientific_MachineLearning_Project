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

v4 : Try with Gpu, transfer all the data to gpu at the beginning in DataLoader
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
batch_sizes = [128,256,512,1024,2048]
# num_workers = [1,2,4] can't use workers if your data is already on gpu (I think)
# pin_memories = [True,False] can't pin if your data is already on gpu

for batch_size in batch_sizes:
    my_dataset = MyDataset()
    my_dataset.load_to_gpu(device)
    loader = DataLoader(dataset = my_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

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
            # X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            y_hat = model.forward(X_batch)
            loss = criterion(y_hat,Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            N_iters += 1
        if i % 10 == 0:
            print(f'epoch {i}, loss = {loss}')

    t1 = time.time() - t0
    print("%d epochs in %.2f seconds -script - v4 BatchSize=%d" %(N_epochs,t1,batch_size))



























exit()
