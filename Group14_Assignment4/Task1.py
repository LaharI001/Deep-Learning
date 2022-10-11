

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from Model import plotLosses,getActualAndPredictedOutput,trainModel,collate_fn
from Model import RNNModel,LSTMModel
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import os
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import copy



paths = "./Handwriting_Data"
ClassName = ['a','ai','bA','dA','tA']
ModelName = ['train','dev']
trainList = [[],[],[],[],[]]
testList  = [[],[],[],[],[]]


def process(string):
    string  = str(string)
    s = ""
    for i in string:
        if(i == "\\"):
            s += '/'
        else:
            s = s + i
    return s

def loadFile():
    for className in ClassName:
        for modelName in ModelName:
            for elem in Path(paths+"/"+className+"/"+modelName).rglob('*.*'):
                idx = ClassName.index(className)
                xx = process(elem)
                if(modelName == 'train'):
                    trainList[idx].append(xx)
                else:
                    testList[idx].append(xx)  

loadFile()


def plot(x, y,z):
    plt.plot(x, y, color='blue', linewidth = 1,
             marker='o', markerfacecolor='blue', markersize=4)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Visualisation of Letter '+z)
    plt.show()



def ListofSample(sample):
    text_file = open(sample, "r")
    data = text_file.read()
    text_file.close()
    data = list(data.split())
    x = []
    y = []
    for i in range(1,int(data[0])*2,2):
        x.append(float(data[i]))
        y.append(float(data[i+1]))
    x = np.array(x)
    y = np.array(y)
    x = (x - np.min(x))/(np.max(x) - np.min(x))
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    return x,y



initial = 0
for i in trainList[0]:
    if (initial == 6):
        initial = 0
        break
    initial = initial + 1
    x,y  = ListofSample(i)
    plot(x,y, ClassName[0])



for i in trainList[1]:
    if (initial == 6):
        initial = 0
        break
    initial = initial + 1
    x,y  = ListofSample(i)
    plot(x,y,ClassName[1])



for i in trainList[2]:
    if (initial == 6):
        initial = 0
        break
    initial = initial + 1
    x,y  = ListofSample(i)
    plot(x,y,ClassName[2])



for i in trainList[3]:
    if (initial == 6):
        initial = 0
        break
    initial = initial + 1
    x,y  = ListofSample(i)
    plot(x,y,ClassName[3])



for i in trainList[4]:
    if (initial == 6):
        initial = 0
        break
    initial = initial + 1
    x,y  = ListofSample(i)
    plot(x,y,ClassName[4])






class HandWriting(Dataset):
    def __init__(self, path: str, train=True, nPlots=2):
        self.nPlots = nPlots
        if train == True:
            subpath = "train"
        else:
            subpath = "dev"
        self.dir = Path(path)
        with open(self.dir/"Mapping.txt") as fhand:
            for i in fhand:
                if i.startswith(f"Group {14}"):
                    line = i
                    break
        chars = self.getChars(line)
        self.numChars = len(chars)
        self.char2int = {chars[i]: i for i in range(len(chars))}
        self.int2char = {i: chars[i] for i in range(len(chars))}
        HandData = []
        plotData = []
        for char in chars:
            ploti = 0
            folder = self.dir/char/subpath
            files = os.listdir(folder)
            for file in files:
                with open(folder/file) as fhand:
                    xy = fhand.read()
                xy = xy.split()
                xy = xy[1:]
                xy = list(map(float, xy))
                xy = np.array(xy)
                xy = torch.from_numpy(xy)
                xy = xy.reshape((-1, 2))
                xy = (xy-xy.min(dim=0)[0])/(xy.max(dim=0)[0]-xy.min(dim=0)[0])
                xy = xy[::5, :]
                xy = xy.float()
                label = torch.tensor(self.char2int[char])
                if ploti < nPlots:
                    plotData.append((xy.detach().numpy(), char))
                ploti += 1
                HandData.append((xy, label))
        self.plotData = plotData
        self.xy = HandData
        self.chars = chars


    def __len__(self) -> int:
        return len(self.xy)

    def __getitem__(self, idx: int):
        return self.xy[idx]

    def getChars(self, line: str):
        colonIndex = line.find(":")
        line = line[colonIndex+1:]
        chars: List = line.split(",")
        chars = list(map(lambda x: x.strip(), chars))
        return chars


trainDataset=HandWriting("./Handwriting_Data", nPlots=5)
testDataset=HandWriting("./Handwriting_Data",train=False, nPlots=5)
trainDataloader=DataLoader(dataset=trainDataset,batch_size=32,shuffle=True,collate_fn=collate_fn)
testDataloader=DataLoader(dataset=testDataset,batch_size=32,shuffle=False,collate_fn=collate_fn)

hidden_layers=[[40], [70], [30,30], [240, 70], [70, 40, 50], [130, 30, 40]]


# ## RNN Model


rnn_train_acc=[]
rnn_test_acc=[]
rnn_epochs=[]
lr_used=[]
hl=[]
for layers in hidden_layers:
    rnn1=RNNModel(layers,2,trainDataset.numChars,len(layers))
    criterion=nn.CrossEntropyLoss()
    for lr in [0.01]:
        print(f"Hidden Layers: {layers}, learning rate: {lr}")
        rnn=copy.deepcopy(rnn1)
        optimizerRNN=torch.optim.Adam(rnn.parameters(),lr=lr)
        threshold=1e-4
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rnn, trainLossesRNN, epochs=trainModel(device,rnn,criterion,optimizerRNN,threshold,trainDataloader,trainDataset)
        plotLosses(trainLossesRNN)

        predTrainRNN,actTrainRNN=getActualAndPredictedOutput(rnn,device,trainDataloader)
        predTestRNN,actTestRNN=getActualAndPredictedOutput(rnn,device,testDataloader)

        rnn_train_acc.append(accuracy_score(actTrainRNN,predTrainRNN))
        rnn_test_acc.append(accuracy_score(actTestRNN,predTestRNN))
        rnn_epochs.append(epochs)
        lr_used.append(lr)
        hl.append(layers)

        cmTrainRNN=confusion_matrix(actTrainRNN,predTrainRNN)
        cmTestRNN=confusion_matrix(actTestRNN,predTestRNN)

        dispTrainRNN=ConfusionMatrixDisplay(cmTrainRNN,display_labels=trainDataset.chars)
        dispTrainRNN.plot()
        plt.title(f"Confusion Matrix for Training data(RNN)")
        plt.show()

        dispTestRNN=ConfusionMatrixDisplay(cmTestRNN,display_labels=testDataset.chars)
        dispTestRNN.plot()
        plt.title(f"Confusion Matrix for Testing data (RNN)")
        plt.show()


# ## LSTM Model


lstm_train_acc=[]
lstm_test_acc=[]
lstm_epochs=[]
lr_used=[]
hl=[]
for layers in hidden_layers:
    lstm1=LSTMModel(layers,2,trainDataset.numChars,len(layers))
    criterion=nn.CrossEntropyLoss()
    for lr in [0.01]:
        lstm=copy.deepcopy(lstm1)
        print(f"Hidden Layers: {layers}, learning rate: {lr}")
        optimizerLSTM=torch.optim.Adam(lstm.parameters(),lr=lr)
        threshold=1e-4
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rnn, trainLossesLSTM, epochs=trainModel(device,lstm,criterion,optimizerLSTM,threshold,trainDataloader,trainDataset)
        plotLosses(trainLossesLSTM)

        predTrainLSTM,actTrainLSTM=getActualAndPredictedOutput(lstm,device,trainDataloader)
        predTestLSTM,actTestLSTM=getActualAndPredictedOutput(lstm,device,testDataloader)

        lstm_train_acc.append(accuracy_score(actTrainLSTM,predTrainLSTM))
        lstm_test_acc.append(accuracy_score(actTestLSTM,predTestLSTM))
        lstm_epochs.append(epochs)
        lr_used.append(lr)
        hl.append(layers)

        cmTrainLSTM=confusion_matrix(actTrainLSTM,predTrainLSTM)
        cmTestLSTM=confusion_matrix(actTestLSTM,predTestLSTM)

        dispTrainLSTM=ConfusionMatrixDisplay(cmTrainLSTM,display_labels=trainDataset.chars)
        dispTrainLSTM.plot()
        plt.title(f"Confusion Matrix for Training data(LSTM)")
        plt.show()

        dispTestLSTM=ConfusionMatrixDisplay(cmTestLSTM,display_labels=testDataset.chars)
        dispTestLSTM.plot()
        plt.title(f"Confusion Matrix for Testing data (LSTM)")
        plt.show()





