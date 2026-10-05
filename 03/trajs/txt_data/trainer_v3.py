import torch
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Dataset,random_split
import time
import argparse
"""
Split the data train/test:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/

Use a wider and shallower network:
https://www.baeldung.com/cs/neural-networks-hidden-layers-criteria
Try 10-15-20 layers with 1,5,10,15k width

For 10 layers 4k seems to be the max width, independent of the batch size
The size of network dominates GPU usage :
https://staff.sharcnet.ca/guanw/GIS-2020-09-23/IsmyNNtoobigforGPU.pdf

"""

class MyDataset(Dataset):
    def __init__(self,filename):
        data = np.loadtxt(filename,dtype=np.float32)
        self.n_samples = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])
        self.y_data = torch.from_numpy(data[:,-1])
        self.y_data = torch.reshape(self.y_data,(self.n_samples,1))

    # def load_to_gpu(self,device):
    #     self.x_data = self.x_data.to(device)
    #     self.y_data = self.y_data.to(device)

    def get_split_counts(self,fraction):
        N_test = self.n_samples*fraction
        N_test = int(N_test)
        N_train = self.n_samples - N_test
        return N_train,N_test

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

parser = argparse.ArgumentParser(description="")
non_opt = parser.add_argument_group("mandatory arguments")
non_opt.add_argument('-i', '--input', metavar="<dat>", type=str, dest="input_data",
required=True, help="data file" )

args = parser.parse_args()
input_data = args.input_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

script_name = sys.argv[0]
script_name = script_name[:-3]

batch_size = 256
width = 5000
depth = 10
lr = 0.001
N_epochs = 1501
target_loss = 0.0001
test_fraction = 0.2

my_dataset = MyDataset(input_data)
# my_dataset.load_to_gpu(device)
N_train, N_test = my_dataset.get_split_counts(test_fraction)

train_dataset, test_dataset = random_split(my_dataset,[N_train,N_test])

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory=True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

data_iter = iter(train_loader)
dX,dY = data_iter.next()
X_shape = dX.size()
input_size = X_shape[-1]
model = TheNet(input_size,width,depth)
model = nn.DataParallel(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr )
criterion = nn.MSELoss()


t0 = time.time()
for i in range(N_epochs):
    train_loss = []
    model.train()
    for j,(X_batch,Y_batch) in enumerate(train_loader):
        # print('bak2')
        # time.sleep(5)
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        # time.sleep(5)
        # print('bak3')
        # time.sleep(5)
        y_hat = model.forward(X_batch)
        # time.sleep(5)
        loss = criterion(y_hat,Y_batch)
        train_loss.append(loss.cpu())
        # print('bak')
        # time.sleep(5)
        optimizer.zero_grad()
        # time.sleep(5)
        # print('bak2')
        # time.sleep(5)
        loss.backward()
        # time.sleep(5)
        # print('bak3')
        print('bak3')
        time.sleep(5)
        optimizer.step()
        time.sleep(15)
        exit()

    if i % 50 == 0:
        model.eval()
        with torch.no_grad():
            ## compute train loss
            train_loss = np.average(np.array(train_loss))
            ## compute test loss
            test_loss = []
            for jj,(X_batch,Y_batch) in enumerate(test_loader):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                y_hat = model.forward(X_batch)
                loss = criterion(y_hat,Y_batch)
                test_loss.append(loss.cpu())
            test_loss = np.average(np.array(test_loss))
            print(f'epoch {i}, train_loss = {train_loss}, test_loss = {test_loss}')

t1 = time.time() - t0

# torch.save(model,'./model_%s_lr%.4f_bs%d_Ne%d.pth'%(script_name,lr,batch_size,N_epochs))
# print("%d epochs in %.2f seconds -script - v5 BatchSize=%d" %(N_epochs,t1,batch_size))




























exit()
