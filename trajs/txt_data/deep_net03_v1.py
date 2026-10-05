import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

"""
What is the best way to load the data to gpu to do the training?
Not so straightforward with minibatche+dataloader

Should we use minibatch?
How many batches? 
"""



class MyDataset(Dataset):
    def __init__(self):
        data = np.loadtxt("./dataset_kirk03.txt",dtype=np.float32)
        self.n_samples = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])
        self.y_data = torch.from_numpy(data[:,-1])

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

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# N_gpu = torch.cuda.device_count()
# gpu_name = torch.cuda.get_device_name(1)
# print(device)
# print(N_gpu)
# print(gpu_name)
# exit()

my_dataset = MyDataset()
loader = DataLoader(dataset = my_dataset, batch_size = 20, shuffle = True, num_workers = 2)

data_iter = iter(loader)
dX,dY = data_iter.next()
X_shape = dX.size()
input_size = X_shape[-1]
model = TheNet(input_size,300,30)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01 )
criterion = nn.MSELoss()

N_epochs = 600
N_iters = 0

for i in range(N_epochs):
    for j,(X_batch,Y_batch) in enumerate(loader):
        y_hat = model.forward(X_batch)
        loss = criterion(y_hat,Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        N_iters += 1
    if i % 10 == 0:
        print(f'epoch {i}, loss = {loss}')


torch.save(model.state_dict(), './model')























exit()
