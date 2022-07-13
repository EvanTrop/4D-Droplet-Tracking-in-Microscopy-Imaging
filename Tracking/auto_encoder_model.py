from ipython_genutils.py3compat import encode
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import numpy as np


class Encoder(nn.Module):

    def __init__(self,inChannels,encodingDim):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels,8,3)
        self.conv2 = nn.Conv2d(8,16,3)
        self.conv3 = nn.Conv2d(16,16,3)
        self.convLayers = nn.ModuleList([self.conv1,self.conv2,self.conv3])

        self.bn1_dec = nn.BatchNorm2d(8)
        self.bn2_dec = nn.BatchNorm2d(16)
        self.bn3_dec = nn.BatchNorm2d(16)
        self.bn_dec = nn.ModuleList([self.bn1_dec,self.bn2_dec,self.bn3_dec])
      
        self.fc1 = nn.Linear(18496,encodingDim)

    def forward(self,x):

        for i in range(len(self.convLayers)):
            x = self.bn_dec[i](self.convLayers[i](x))
            x = nn.functional.relu(x)   

        x = self.fc1(x.flatten(start_dim=1))

        return x

class Decoder(nn.Module):

    def __init__(self,inChannels,encodingDim):
        super().__init__()

        self.fc2 = nn.Linear(encodingDim,18496)

        self.deconv1 = nn.ConvTranspose2d(16,16,3)
        self.deconv2 = nn.ConvTranspose2d(16,8,3)
        self.deconv3 = nn.ConvTranspose2d(8,1,3)
        self.deconvLayers = nn.ModuleList([self.deconv1,self.deconv2,self.deconv3])
        
        self.bn1_inc = nn.BatchNorm2d(16)
        self.bn2_inc = nn.BatchNorm2d(8)
        self.bn3_inc = nn.BatchNorm2d(1)
        self.bn_inc = nn.ModuleList([self.bn1_inc,self.bn2_inc,self.bn3_inc])


    def forward(self,x):

        x = self.fc2(x).reshape(-1,16,34,34)

        for i in range(len(self.deconvLayers)):
            x = self.bn_inc[i](self.deconvLayers[i](x))
            x = nn.functional.relu(x)    

        return x


class AutoEncoder(nn.Module):

    def __init__(self,inChannels,encodingDim):
        super().__init__()
        
        self.encoder = Encoder(inChannels,encodingDim)
        self.decoder = Decoder(inChannels,encodingDim)

        self.loss = nn.L1Loss()

    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x) 

        return x


class DropletImageData(Dataset):

    def __init__(self,images,maxDim):
        super().__init__()

        self.X = images
        self.maxDim = maxDim

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):

        h , w = self.X[index].shape
        w = self.maxDim - w
        h  = self.maxDim - h
        assert(w >=0 and h >= 0)

        w1 = int(w /2)
        h1 = int(h /2)
        img = self.X[index]
        
        if  w % 2 != 0:
          w2 = w1 + 1
        else:
          w2 = w1
        if  h % 2 != 0:
          h2 = h1 + 1
        else:
          h2 = h1

        return np.pad(img,((h1,h2),(w1,w2))).astype("float32")

def train_one_epoch(trainLoader,model,optim,device):
    """
    Training loop for a single epoch, (ie pass through the entire train dataset)

    Params:
        trainLoader: dataloader with trainDataset, X -(W x H images)
        model: GNN model instance
        optim: optimizer instance holding the model arg's parameters
        device: the torch.device to carry out computation
    Returns:
        trainLoss: average training loss across all batches (float)
    """

    #Save gradients
    model.train()

    trainLoss = 0

    #Iterate over batches in loader
    for X in trainLoader:

        optim.zero_grad()

        #move data to specified device
        X = X.to(device)
        X = X.unsqueeze(1)

        #forward pass
        Xhat = model(X)
        
        #calculate loss w/ class weights
        loss = model.loss(Xhat,X)
        trainLoss += loss.item()

        #backward pass and update parameters
        loss.backward()
        optim.step()

    #average the loss across batches
    trainLoss /= len(trainLoader) 

    return trainLoss


def eval_one_epoch(evalLoader,model,device):
    """
    Validation loop for a single epoch during training

    Params:
        evalLoader: dataloader with evalDataset, X - (W x H images)
        model:  model instance
        device: the torch.device to carry out computation
    Returns:
        evalLoss: average validation loss across all batches (float)
        evalAcc: average validation accuracy across all batches (float)
    """
    
    #no gradients needed
    model.eval()

    evalLoss = 0
    evalAcc = 0

    #iterate over the batches in loader
    for X in evalLoader:

        #move data to specified device
        X = X.to(device)
        X = X.unsqueeze(1)

        #forward pass, these are only logits
        Xhat = model(X)

        #calculate loss w/ class weights
        loss = model.loss(Xhat,X)
        evalLoss += loss.item()
        

    # evalAcc /= len(evalLoader)
    evalLoss /= len(evalLoader)
    
    return evalLoss