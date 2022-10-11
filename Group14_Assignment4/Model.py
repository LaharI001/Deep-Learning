from torch import nn
import torch

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class RNNModel(nn.Module):
    def __init__(self, stateDimension, featureDim: int, nOutputFeatures: int, numLayers=2):
        super().__init__()
        self.nFeature = featureDim
        self.nState = stateDimension
        self.nLayers = numLayers
        self.layers=nn.ModuleList([])
        self.nOutput = nOutputFeatures
        for i in range(self.nLayers):
            if i==0:
                self.layers.append(nn.RNN(self.nFeature, self.nState[i],
                          1, batch_first=True))
            else:
                 self.layers.append(nn.RNN(self.nState[i-1], self.nState[i],
                          1, batch_first=True))
            layer_name="layer"+str(i+1)
            self.register_buffer(layer_name, torch.randn(1, 1, self.nState[i]))
        self.fcnn = nn.Linear(self.nState[-1], self.nOutput)

    def forward(self, x: torch.Tensor):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            bs = x.batch_sizes.max().item()
        else:
            bs = x.shape[0]
        layer1 = self.layer1.repeat(1, bs, 1)
        res, op = self.layers[0](x, layer1)
        for i in range(1, self.nLayers):
            layer_name="layer"+str(i+1)
            layer = self.get_buffer(layer_name).repeat(1, bs, 1)
            res, op = self.layers[i](res, layer)
        op = op[-1]
        output = self.fcnn(op)
        return output


class LSTMModel(nn.Module):
    def __init__(self, stateDimension, featureDim: int, nOutputFeatures: int, numLayers=2):
        super().__init__()
        self.nFeature = featureDim
        self.nState = stateDimension
        self.nLayers = numLayers
        self.layers=nn.ModuleList([])
        self.nOutput = nOutputFeatures
        for i in range(self.nLayers):
            if i==0:
                self.layers.append(nn.LSTM(self.nFeature, self.nState[i],
                          1, batch_first=True))
            else:
                 self.layers.append(nn.LSTM(self.nState[i-1], self.nState[i],
                          1, batch_first=True))
            l="l"+str(i+1)
            c="c"+str(i+1)
            self.register_buffer(l, torch.randn(1, 1, self.nState[i]))
            self.register_buffer(c, torch.randn(1, 1, self.nState[i]))
        self.fcnn = nn.Linear(self.nState[-1], self.nOutput)

    def forward(self, x: torch.Tensor):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            bs = x.batch_sizes.max().item()
        else:
            bs = x.shape[0]
        l1 = self.l1.repeat(1, bs, 1)
        c1 = self.c1.repeat(1, bs, 1)
        res, (op1, op2) = self.layers[0](x, (l1, c1))
        for i in range(1, self.nLayers):
            ln="l"+str(i+1)
            cn="c"+str(i+1)
            l = self.get_buffer(ln).repeat(1, bs, 1)
            c = self.get_buffer(cn).repeat(1, bs, 1)
            res, (op1, op2) = self.layers[i](res, (l, c))
        op1 = op1[-1]
        output = self.fcnn(op1)
        return output
    
    

def plotLosses(trainLosses):
    xs = np.arange(0, len(trainLosses))
    plt.plot(xs, trainLosses, label="Train Loss")
    plt.title("Cross Entropy Loss vs No. of epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.show()


def getActualAndPredictedOutput(model, device, dataloader) -> Tuple[List, List]:
    pred = []
    act = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            yhat = model(x)
            _, ypred = yhat.max(dim=1)
            ypred = ypred.detach().cpu().numpy()
            ypred = list(ypred)
            y = list(y.detach().cpu().numpy())
            pred.extend(ypred)
            act.extend(y)
    return pred, act


def trainModel(device, model, criterion, optimizer, threshold, trainDataloader, trainDataset, verbose=True,):
    rnn = model.to(device)
    epoch = 0
    trainLoss = 0
    prevTrainLoss = 0
    trainLosses = []
    while True:
        epoch += 1
        trainLoss = 0
        rnn.train()
        if verbose:
            print()
            print(f"Epoch #{epoch} {'-'*30}")
        for x, y in trainDataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            pred = rnn(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if type(x) == torch.nn.utils.rnn.PackedSequence:
                nInput = x.batch_sizes.max().item()
            else:
                nInput = x.shape[0]
            trainLoss += nInput*loss.item()
        trainLoss = trainLoss/len(trainDataset)
        trainLosses.append(trainLoss)
        if verbose:
            print(f"Train Loss: {trainLoss}")
            print(f"Previous Training Loss: {prevTrainLoss}")
        rnn.eval()

        if ((abs(trainLoss-prevTrainLoss) < threshold) and prevTrainLoss != 0) or epoch>1000:
            break
        else:
            prevTrainLoss = trainLoss
    if verbose:
        print(
            f"Training continued till {epoch} number of epochs.\n The final training loss being {trainLoss}.")
    return rnn, trainLosses, epoch

def collate_fn(listOfData):
    x = list(map(lambda x: x[0], listOfData))
    y = list(map(lambda x: x[1], listOfData))
    xxs = nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    yys = torch.stack(y)
    return xxs, yys
