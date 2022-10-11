import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import os
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from Model import plotLosses,getActualAndPredictedOutput,trainModel,collate_fn
from Model import RNNModel,LSTMModel
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import pandas as pd
import copy



class CV(Dataset):
    def __init__(self, path: str, train=True):
        if train == True:
            subpath = "Train"
        else:
            subpath = "Test"
        self.dir = Path(path)
        with open(self.dir/"Mapping_CV.txt") as fhand:
            for i in fhand:
                if i.startswith(f"Group-{14}"):
                    line = i
                    break
        cvs = self.getCVs(line)
        self.numcvs = len(cvs)
        self.char2int = {cvs[i]: i for i in range(len(cvs))}
        self.int2char = {i: cvs[i] for i in range(len(cvs))}
        CVData = []
        for cv in cvs:
            folder = self.dir/cv/subpath
            files = os.listdir(folder)
            for file in files:
                with open(folder/file) as fhand:
                    xy = fhand.readlines()
                xy = list(map(lambda x: list(map(float, x.split())),xy))
                xy = np.array(xy)
                xy = torch.from_numpy(xy)
                xy = xy.float()
                label = torch.tensor(self.char2int[cv])
                CVData.append((xy, label))
        self.xy = CVData
        self.chars = cvs

    def __len__(self) -> int:
        return len(self.xy)

    def __getitem__(self, idx: int):
        return self.xy[idx]

    def getCVs(self, line: str):
        gText = f"Group-{14}"
        groupIndex = line.find(gText)
        line = line[groupIndex+len(gText)+1:]
        cvs: List = line.split(",")
        cvs = list(map(lambda x: x.strip(), cvs))
        return cvs



trainDataset=CV("./CV_Data")
testDataset=CV("./CV_Data",train=False)
trainDataloader=DataLoader(dataset=trainDataset,batch_size=32,shuffle=True,collate_fn=collate_fn)
testDataloader=DataLoader(dataset=testDataset,batch_size=32,shuffle=False,collate_fn=collate_fn)
hidden_layers=[[40], [70], [30,30], [240, 70], [70, 40, 50],[130, 30, 40]]

rnn_train_acc=[]
rnn_test_acc=[]
rnn_epochs=[]
lr_used=[]
hl=[]
for layers in hidden_layers:
    rnn1=RNNModel(layers,39,trainDataset.numcvs,len(layers))
    criterion=nn.CrossEntropyLoss()
    for lr in [0.01]:
        rnn=copy.deepcopy(rnn1)
        print(f"Hidden Layers: {layers}, learning rate: {lr}")
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
    lstm1=LSTMModel(layers,39,trainDataset.numcvs,len(layers))
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
