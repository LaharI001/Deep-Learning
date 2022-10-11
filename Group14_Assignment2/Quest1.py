#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from PIL import Image


# In[2]:


def load_images_from_folder(folder):
    images = []
    y = []
#     d = {0:0,2:1, 3:2, 7:3, 8:4}
    for i in folder:
        for filename in os.listdir(i):
#             print(filename)
            img = Image.open(os.path.join(i,filename))
            if img is not None:
                images.append(np.asarray(img))
            # print(int(i[-1]), end="")
            y.append(int(i[-1]))
    images = np.array(images)
    y = np.array(y)
    x_data , y_data = shuffle (images, y)
    return x_data, y_data
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'blue' if val > 90 else 'black'
    return 'color: % s' % color


# In[3]:


train_data, y_train = load_images_from_folder(["./Group_14/train/0", "./Group_14/train/2", "./Group_14/train/3", "./Group_14/train/7", "./Group_14/train/8"])
test_data, y_test = load_images_from_folder(["./Group_14/test/0", "./Group_14/test/2", "./Group_14/test/3", "./Group_14/test/7", "./Group_14/test/8"])
val_data, y_val = load_images_from_folder(["./Group_14/val/0", "./Group_14/val/2", "./Group_14/val/3", "./Group_14/val/7", "./Group_14/val/8"])
train_data = train_data/255
val_data = val_data/255
test_data = test_data/255
plt.imshow(train_data[0])
plt.show()
print("y_train[0]: "+str(y_train[0]))


# In[4]:


from fcnn import fcnn

optimizers = ["SGD", "VGD","NAG","RMSProp","ADM"]
keras_optimizers= [tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
      tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0),
      tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
      tf.keras.optimizers.RMSprop(learning_rate=0.0001,rho=0.90, momentum=0.9, epsilon=1e-07),
      tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True)]
batch = [1,len(train_data),32,32,32]
layers_matrix = np.array([[200,100,50],[512,128,32],[300,400,100],[800,300,50],[400,150,60]])
epoch_each_arch = []
best_arch = {}


# In[7]:


def call_fcnn(optimizer_name, keras_optimizer,batch):
    
#     temp = []
    tabulate_epoch = []
    training_acc = []
    val_acc = []
    model_val_acc = 0
    opt_epoch = []
    for i in layers_matrix:
        print(i[0],i[1], i[2])
        fcnn_model = fcnn([28,28],i,len(np.unique(y_train)), keras_optimizer, True)
        fcnn_model._fit_(np.array(train_data), np.array(y_train), epochs=1000, batch_size=batch)
#         temp.append(fcnn_model.end_epoch)
        fcnn_model.plot_err_vs_epoch(optimizer_name)
        opt_epoch.append(fcnn_model.end_epoch)
        
        pred = fcnn_model.prediction(train_data)
        trainAcc = fcnn_model.accuracy(pred, y_train)
        training_acc.append(trainAcc)
        print("Training Accuracy:", trainAcc)
        fcnn_model.confusion_matrix(pred, y_train)

        pred = fcnn_model.prediction(val_data)
        valAcc = fcnn_model.accuracy(pred, y_val)
        val_acc.append(valAcc)
        print("Validation Accuracy:", valAcc)
        fcnn_model.confusion_matrix(pred, y_val)
        # train_acc = fcnn_model.accuracy(pred)
        
        
        if valAcc>model_val_acc:
            model_val_acc = valAcc
            best_arch[optimizer_name] = fcnn_model
    epoch_each_arch.append(opt_epoch) 
    df = pd.DataFrame(
        {"layer 1 neurons": layers_matrix[:,0],
        "layer 2 neurons": layers_matrix[:,1],
        "layer 3 neurons": layers_matrix[:,2],
        "Training Accuracy": training_acc,
        "Validation Accuracy": val_acc}
    )
    print(df)
    df.style.applymap(color_negative_red)


# In[8]:


call_fcnn(optimizers[0],keras_optimizers[0],batch[0])


# In[9]:


call_fcnn(optimizers[1],keras_optimizers[1],batch[1])


# In[10]:


call_fcnn(optimizers[2],keras_optimizers[2],batch[2])


# In[11]:


call_fcnn(optimizers[3],keras_optimizers[3],batch[3])


# In[12]:


call_fcnn(optimizers[4],keras_optimizers[4],batch[4])


# In[13]:


print(best_arch)


# In[14]:


df = pd.DataFrame(
    {"layer 1 neurons": layers_matrix[:,0],
    "layer 2 neurons": layers_matrix[:,1],
    "layer 3 neurons": layers_matrix[:,2],
    "SGD": epoch_each_arch[0],
    "VGD": epoch_each_arch[1],
    "NAG":  epoch_each_arch[2],
    "RMSProp":  epoch_each_arch[3],
    "Adam": epoch_each_arch[4]}
)
print(df)


# In[21]:


pred = best_arch["SGD"].prediction(test_data)
best_arch["SGD"].confusion_matrix(pred, y_test)
test_acc = best_arch["SGD"].accuracy(pred, y_test)
print("Test Accuracy:", test_acc)
# pred = best_arch["SGD"].prediction(train_data)
# best_arch["SGD"].confusion_matrix(pred, y_train)
# test_acc = best_arch["SGD"].accuracy(pred, y_train)
# print("Test Accuracy:", test_acc)


# In[17]:


pred = best_arch["VGD"].prediction(test_data)
best_arch["VGD"].confusion_matrix(pred, y_test)
test_acc = best_arch["VGD"].accuracy(pred, y_test)
print("Test Accuracy:", test_acc)


# In[18]:


pred = best_arch["NAG"].prediction(test_data)
best_arch["NAG"].confusion_matrix(pred, y_test)
test_acc = best_arch["NAG"].accuracy(pred, y_test)
print("Test Accuracy:", test_acc)


# In[19]:


pred = best_arch["RMSProp"].prediction(test_data)
best_arch["RMSProp"].confusion_matrix(pred, y_test)
test_acc = best_arch["RMSProp"].accuracy(pred, y_test)
print("Test Accuracy:", test_acc)


# In[20]:


pred = best_arch["ADM"].prediction(test_data)
best_arch["ADM"].confusion_matrix(pred, y_test)
test_acc = best_arch["ADM"].accuracy(pred, y_test)
print("Test Accuracy:", test_acc)


# In[ ]:




