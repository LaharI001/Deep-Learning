

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.models import Model, Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from numpy import expand_dims
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt

import torch
from torch.nn import ReLU

from torch.autograd import Variable
from torchvision import models

path = "./files"

class VGG_19:
  def __init__(self,path):
    self.path=path
    self.build()

  def build(self):
    self.vgg19 = VGG19(weights='imagenet')
    image = self.vgg19.layers[0].output
    out = self.vgg19.get_layer("flatten").output
    self.vgg19 = Model(inputs=image,outputs=out)
    self.vgg19.summary()
    
    self.train = pd.read_csv(self.path+"/train.csv")
    self.trainLabel = pd.read_csv(self.path+"/trainLabel.csv")
    self.test = pd.read_csv(self.path+"/test.csv")
    self.testLabel = pd.read_csv(self.path+"/testLabel.csv")
    self.val = pd.read_csv(self.path+"/val.csv")
    self.valLabel = pd.read_csv(self.path+"/valLabel.csv")
    print(self.train)
    print(self.trainLabel)
    print(self.test)
    print(self.testLabel)
    print(self.val)
    print(self.valLabel)
    self.model = Sequential()
    self.model.add(Dense(4096, activation="relu", name='fc1'))
    self.model.add(Dense(4099, activation="relu", name='fc2'))
    self.model.add(Dense(3, activation="softmax", name='predictions'))
    self.model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    self.model.fit(self.train,self.trainLabel, batch_size=32, epochs=5,validation_data=(self.val,self.valLabel), verbose=0)



  def evaluate(self):
    trainLabel = [list(i).index(max(list(i))) for i in self.trainLabel.values]
    predTrain = [list(i).index(max(list(i))) for i in self.model.predict(self.train.values)]
    valLabel = [list(i).index(max(list(i))) for i in self.valLabel.values]
    predVal = [list(i).index(max(list(i))) for i in self.model.predict(self.val.values)]
    testLabel = [list(i).index(max(list(i))) for i in self.testLabel.values]
    predTest = [list(i).index(max(list(i))) for i in self.model.predict(self.test.values)]


  def visualise(self,path,name):
    env={
        'grand_piano':[7,15,18,32,52],
         'starfish':[6,20,31,36,59],
         'trilobite':[2,29,32,50,59]
    }
    model = self.vgg19
    model = Model(inputs=model.inputs,outputs=model.layers[1].output)
    img = load_img(self.path+path,target_size=(224,224))
    plt.imshow(img)
    plt.show()
    img = img_to_array(img)
    img = expand_dims(img,axis=0)
    img = preprocess_input(img)
    feature_maps = model.predict(img)
    
    for i in env[name]:
      plt.imshow(feature_maps[0,:,:,i],cmap="gray")
      plt.show()

  def guided_backpropagate(self, path, name):
    img = load_img(self.path+path,target_size=(224,224))
    plt.imshow(img)
    plt.show()
    enc={
        "grand_piano": 286,
         "starfish": 725,
         "trilobite": 104
    }
    orgImage = Image.open(self.path+path).convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    orgImage = orgImage.resize((224,224), Image.ANTIALIAS)
    imgArr = np.float32(orgImage)
    imgArr = imgArr.transpose(2,0,1)
    for i, _ in enumerate(imgArr):
      imgArr[i]/=255
      imgArr[i]-=mean[i]
      imgArr[i]/=std[i]
    imgTen = torch.from_numpy(imgArr).float()
    imgTen.unsqueeze_(0)
    prepImage = Variable(imgTen,requires_grad=True)
    preTrainModel = models.vgg19(pretrained=True)
    preTrainModel.eval()
    fro = []
    global grad
    grad = None
    def rbhf(modu, gIn, gOut):
      cfo = fro[-1]
      cfo[cfo>0]=1
      mgo = cfo*torch.clamp(gIn[0],min=0.0)
      del fro[-1]
      return (mgo,)
    def rfhf(modu, tIn, tOut):
      fro.append(tOut)
    for i,modu in preTrainModel.features._modules.items():
      if isinstance(modu, ReLU):
        modu.register_backward_hook(rbhf)
        modu.register_forward_hook(rfhf)
    def hf(modu,gIn,gOut):
      global grad
      grad = gIn[0]
    fl = list(preTrainModel.features._modules.items())[1][1]
    fl.register_backward_hook(hf)
    output = preTrainModel(prepImage)
    preTrainModel.zero_grad()
    oneHot = torch.FloatTensor(1,output.size()[-1]).zero_()
    oneHot[0][enc[name]]=1
    output.backward(gradient=oneHot)
    img = grad.data.numpy()[0]
    img = img - img.min()
    img /= img.max()
    
    for i in range(0,60,12):
      img[i] = img[i] - img[i].min()
      img[i]/=img[i].max()
      temp = np.array([img[i],img[i],img[i]])
      temp=temp.transpose(1,2,0)+np.full((224,224,3),0.9)
      
      temp/=temp.max()
      plt.imshow(temp)
      plt.show()

obj = VGG_19(path)
obj.evaluate()
print("grand_piano Imaage")
obj.visualise("/train/grand_piano/image_0001.jpg","grand_piano")
print("starfish Imaage")
obj.visualise("/train/starfish/image_0002.jpg","starfish")
print("trilobite Imaage")
obj.visualise("/train/trilobite/image_0001.jpg","trilobite")
print("grand_piano Imaage")
obj.guided_backpropagate("/train/grand_piano/image_0001.jpg","grand_piano")
print("starfish Imaage")
obj.guided_backpropagate("/train/starfish/image_0002.jpg","starfish")
print("trilobite Imaage")
obj.guided_backpropagate("/train/trilobite/image_0001.jpg","trilobite")

