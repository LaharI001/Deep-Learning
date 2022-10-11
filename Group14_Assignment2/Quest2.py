#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
# import imageio
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
    x_data , y_data = shuffle(images, y)
    return x_data, y_data


# In[3]:


train_data, y_train = load_images_from_folder(["./Group_14/train/0", "./Group_14/train/2", "./Group_14/train/3", "./Group_14/train/7", "./Group_14/train/8"])
test_data, y_test = load_images_from_folder(["./Group_14/test/0", "./Group_14/test/2", "./Group_14/test/3", "./Group_14/test/7", "./Group_14/test/8"])
val_data, y_val = load_images_from_folder(["./Group_14/val/0", "./Group_14/val/2", "./Group_14/val/3", "./Group_14/val/7", "./Group_14/val/8"])
train_data = train_data/255
val_data = val_data/255
test_data = test_data/255


# In[4]:



def MSE(pred, actual):
    pred = pred.reshape(len(pred), 28*28)
    actual = actual.reshape(len(actual), 28*28)
    # print(pred[0])
    # print(actual[0])
    mse = 0
    for i in range(len(pred)):
        t = np.square(pred[i]-actual[i])
        mse += np.sum(t,axis=0)
    return mse/len(pred)


# In[5]:



from autoencoder import Autoencoder


class ModelCallBack(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.avg_error = []
        self.prev_error = 0
        self.current_error = 0
        self.end_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        if epoch == 0:
            self.prev_error = logs["mean_squared_error"]
            self.current_error = logs["mean_squared_error"]
        else:
            self.current_error = logs["mean_squared_error"]
            if abs(self.current_error-self.prev_error)<1e-4:
                # self.end_epoch = epoch
                self.model.stop_training = True
            else:
                self.prev_error = self.current_error
        # print(logs["sparse_categorical_crossentropy"])
        print("Epoch", epoch, "done | MSE:", logs["mean_squared_error"])
        self.end_epoch = epoch
        self.avg_error.append(self.prev_error)


# In[6]:


best_arch = {}


# In[10]:





def run_q2(AE_arch):

    train_RE = []
    val_RE = []
    val_best_arch = 1000
    
    for i in range(len(AE_arch)):
        print("-"*25+"Training Autoencoder with Architecture:"+str(AE_arch[i])+"-"*25)
        model_callback = ModelCallBack()
        AE = Autoencoder(AE_arch[i])
        AE.compile(optimizer='adam', loss="mean_squared_error", metrics=['mean_squared_error'])

        AE.fit(np.array(train_data), 
               np.array(train_data), 
               epochs = 1000, 
               shuffle=True, 
               verbose=0,
               callbacks=[model_callback])

        AE.avg_error = model_callback.avg_error
        AE.end_epoch = model_callback.end_epoch

        train_pred = AE.predict(train_data)
        val_pred = AE.predict(val_data)

        train_RE.append(MSE(train_pred, train_data))
        print("Training Reconstruction Error:", train_RE[-1])
        val_RE.append(MSE(val_pred, val_data))
        print("Validation Reconstruction Error:", val_RE[-1])
        if val_RE[-1]<val_best_arch :
            best_arch["Hidden Layer {}".format(len(AE_arch[0]))] = AE
            val_best_arch = val_RE[-1]
        print()

    data = {}
    for j in range(len(AE_arch[0])):
        data["Layer {} Neurons".format(j+1)] = AE_arch[:,j]  

    data["Training Reconstruction Error"] = train_RE 
    data["Validation Reconstruction Error"] = val_RE
    df = pd.DataFrame(data)
    print(df)


# In[11]:


AE_h1 = np.array([[128], [64], [100]])  # Different Autoencoder with One Hidden Layer 
AE_h3 = np.array([[128,64,128], [512,128,512], [200,100,200]]) # Different Autoencoder with Three Hidden Layer


# In[12]:


run_q2(AE_h1)


# In[14]:


run_q2(AE_h3)


# In[15]:


print(best_arch)


# In[16]:


best_arch["Hidden Layer 1"].hidden_layers


# In[17]:


best_arch["Hidden Layer 3"].hidden_layers


# In[18]:


pred = best_arch["Hidden Layer 1"].predict(test_data) 
print("Test Reconstruction Error of Best Architecture with 1 Hidden Layer:", MSE(pred, test_data))
pred = best_arch["Hidden Layer 3"].predict(test_data) 
print("Test Reconstruction Error of Best Architecture with 3 Hidden Layer:", MSE(pred, test_data))


# In[19]:


def plot_error_vs_epoch(best_arch):
    plt.plot(range(best_arch.end_epoch+1), best_arch.avg_error)
    plt.xlabel("No. of Epochs")
    plt.ylabel("Average Errors")
    plt.title("Average Error Vs Epoch of Autoencoder with \n{} Hidden Layer :{}".format(len(best_arch.hidden_layers), best_arch.hidden_layers))
    plt.show()


# In[20]:


print("-"*20+"Plot of Avg Error v/s Epoch of Best Architecture with 1 Hidden Layer"+"-"*20)
plot_error_vs_epoch(best_arch["Hidden Layer 1"])


# In[21]:


print("-"*20+"Plot of Avg Error v/s Epoch of Best Architecture with 3 Hidden Layer"+"-"*20)
plot_error_vs_epoch(best_arch["Hidden Layer 3"])


# In[22]:


import random

def plot_Re_and_orig_images(best_arch, data, n):
     
    randomlist = random.sample(range(len(data)), n)
    encoded_imgs = best_arch.encoder(train_data).numpy()
    decoded_imgs = best_arch.decoder(encoded_imgs).numpy()
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(train_data[randomlist[i]])
        plt.title("original")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[randomlist[i]])
        plt.title("reconstructed")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return randomlist


# In[23]:


print(plt.style.available)


# In[24]:


print("-"*5+"Plot of Original V/s Reconstructed Error for Training Data of Best Architecture with 1 Hidden Layer"+"-"*5)
h1_rl_train = plot_Re_and_orig_images(best_arch["Hidden Layer 1"], train_data, 8)


# In[25]:


print("-"*5+"Plot of Original V/s Reconstructed Error for validation Data of Best Architecture with 1 Hidden Layer"+"-"*5)
h1_rl_val = plot_Re_and_orig_images(best_arch["Hidden Layer 1"], val_data, 8)


# In[26]:


print("-"*5+"Plot of Original V/s Reconstructed Error for Training Data of Best Architecture with 3 Hidden Layer"+"-"*5)
h3_rl_train = plot_Re_and_orig_images(best_arch["Hidden Layer 3"], train_data, 8)


# In[27]:


print("-"*5+"Plot of Original V/s Reconstructed Error for Training Data of Best Architecture with 3 Hidden Layer"+"-"*5)
h3_rl_val = plot_Re_and_orig_images(best_arch["Hidden Layer 3"], val_data, 8)


# In[ ]:





# In[28]:



from fcnn import fcnn

def fcnn_on_compressed(best_arch):

    encoded_train_imgs = best_arch.encoder(train_data).numpy()
    encoded_test_imgs = best_arch.encoder(test_data).numpy()
    encoded_val_imgs = best_arch.encoder(val_data).numpy()

    fcnn_arch = [[64], [64,32,16],[128,32]]
    best_arch_layers = best_arch.hidden_layers
    input = best_arch_layers[len(best_arch_layers)//2]
#     print("fcnn Input:", input)
    fcnn_train_acc = []
    fcnn_test_acc = []
    fcnn_val_acc = []
    for arch in fcnn_arch:
        print("-"*25+"FCNN with Architecture: "+str(arch)+"-"*25)
        fcnn_model = fcnn((input,), arch, 5, "adam" ,False)
        fcnn_model._fit_(np.array(encoded_train_imgs), np.array(y_train))
        pred = fcnn_model.prediction(encoded_train_imgs)
        print("Confusion Matrix of Training Data")
        fcnn_model.confusion_matrix(pred, y_train)
        fcnn_train_acc.append(fcnn_model.accuracy(pred, y_train))

        pred = fcnn_model.prediction(encoded_val_imgs)
        print("Confusion Matrix of Validation Data")
        fcnn_model.confusion_matrix(pred, y_val)
        fcnn_val_acc.append(fcnn_model.accuracy(pred, y_val))
        
        pred = fcnn_model.prediction(encoded_test_imgs)
        print("Confusion Matrix of Test Data")
        fcnn_model.confusion_matrix(pred, y_test)
        fcnn_test_acc.append(fcnn_model.accuracy(pred, y_test))

        

    df = pd.DataFrame({
        "FCNN Architecture": fcnn_arch,
        "Training Accuracy": fcnn_train_acc,
        "Test Accuracy": fcnn_test_acc,
        "Validation Accuracy": fcnn_val_acc
    })
    print(df)


# In[29]:


print("-"*10+"Classification using the compressed representation from the encoder with 1 Hidden Layers:"+"-"*10)
fcnn_on_compressed(best_arch["Hidden Layer 1"])


# In[30]:


print("-"*10+"Classification using the compressed representation from the encoder with 3 Hidden Layers:"+"-"*10)
fcnn_on_compressed(best_arch["Hidden Layer 3"])


# In[31]:


def weight_visualization(best_arch):

    weights = best_arch.layers[0].get_weights()[0]
    print("Weights Shape:",weights.shape)
    weights = weights.T
    i = 0
    k = 0
    
    # t = 0
    for i in range(32):
        plt.figure(figsize=(20, 4))
        j = 4*i
        for k in range(4):
            plt.subplot(1,4,k+1)
            plt.imshow(weights[j+k].reshape((28,28)))
            plt.title("Neuron {}".format(j+k+1))
            # t+=1
        plt.show()


# In[32]:


weight_visualization(best_arch["Hidden Layer 1"])


# In[33]:


# train_data.shape


# In[103]:


class Noise(tf.keras.layers.Layer):
    def __init__(self, mean=0, stddev=1.0, p = 0.2, *args, **kwargs):
        super(Noise, self).__init__(*args, **kwargs)
        self.mean = mean
        self.stddev = stddev
        self.p = p

    def call(self, inputs, 
             training=False # Only add noise in training!
             ):
        # print(inputs.shape)
        random_number = tf.random.uniform((1, ), minval = 0, maxval = 1)[0]
        if training and random_number >= self.p:
            return inputs + tf.random.normal(
                tf.shape(inputs), 
                mean=self.mean,
                stddev=self.stddev
            )
        else:
            return inputs


# In[155]:


random_number = tf.random.uniform((1, ), minval = 0, maxval = 1)[0]
print(random_number)


# In[106]:



bottle_neck_dim = 128

autoencoder_noise = tf.keras.Sequential([
     tf.keras.Input((28, 28)),
     tf.keras.layers.Flatten(),
     Noise(stddev = 0.01, p=0.2),
     tf.keras.layers.Dense(bottle_neck_dim, activation = "sigmoid"),
     tf.keras.layers.Dense(784, activation = "sigmoid"),
    tf.keras.layers.Reshape((28, 28))
    ])

autoencoder_noise.summary()


# In[107]:


input_img = tf.keras.layers.Input(shape=(28,28))
noisy_input = tf.keras.Model(input_img, autoencoder_noise.layers[1](input_img))

# Encoder 
encoder = tf.keras.models.Sequential(
    [
     input_img,
     tf.keras.layers.Flatten(),
     autoencoder_noise.layers[1], 
     autoencoder_noise.layers[2]
    ]
)
encoder.summary()

# Decoder
encoding_dim = tf.keras.Input((bottle_neck_dim, ))
decoder = tf.keras.models.Sequential([
    encoding_dim,
    autoencoder_noise.layers[-2],
    autoencoder_noise.layers[-1]
])

decoder.summary()


# In[108]:


myCallBack = ModelCallBack()
autoencoder_noise.compile(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-08),
              loss="mean_squared_error", metrics = ["mean_squared_error"])
autoencoder_noise.fit(train_data, train_data, epochs=1000, batch_size=32, verbose=0, callbacks=[myCallBack])


# In[109]:


def plot_AE_Re_and_orig_images(best_arch, data, randomlist):
     
#     randomlist = random.sample(range(len(data)), n)
    n = len(randomlist)
    plt.figure(figsize=(20, 4))
    t = 0
    for i in randomlist:
        encoded_imgs = encoder(train_data[i].reshape((1, 28, 28))).numpy()
        decoded_imgs = decoder(encoded_imgs).numpy()
        # display original
        ax = plt.subplot(2, n, t + 1)
        plt.imshow(train_data[i])
        plt.title("original")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, t + 1 + n)
        plt.imshow(decoded_imgs.reshape((28,28)))
        plt.title("reconstructed")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        t += 1
    plt.show()
#     return randomlist


# In[110]:


plot_AE_Re_and_orig_images(autoencoder_noise, train_data, h1_rl_train)


# In[125]:


plot_AE_Re_and_orig_images(autoencoder_noise, val_data, h1_rl_val)


# In[130]:



from fcnn import fcnn

def fcnn_on_compressed():

    encoded_train_imgs =encoder(train_data).numpy()
    encoded_test_imgs = encoder(test_data).numpy()
    encoded_val_imgs = encoder(val_data).numpy()

    fcnn_arch = [[64], [64,32,16],[128,32], [64,32], [32]]
#     best_arch_layers = best_arch.hidden_layers
#     input = best_arch_layers[len(best_arch_layers)//2]
#     print("fcnn Input:", input)
    fcnn_train_acc = []
    fcnn_test_acc = []
    fcnn_val_acc = []
    for arch in fcnn_arch:
        print("-"*25+"FCNN with Architecture: "+str(arch)+"-"*25)
        fcnn_model = fcnn((bottle_neck_dim,), arch, 5, "adam" ,False)
        fcnn_model._fit_(np.array(encoded_train_imgs), np.array(y_train))
        pred = fcnn_model.prediction(encoded_train_imgs)
        print("Confusion Matrix of Training Data")
        fcnn_model.confusion_matrix(pred, y_train)
        fcnn_train_acc.append(fcnn_model.accuracy(pred, y_train))

        pred = fcnn_model.prediction(encoded_val_imgs)
        print("Confusion Matrix of Validation Data")
        fcnn_model.confusion_matrix(pred, y_val)
        fcnn_val_acc.append(fcnn_model.accuracy(pred, y_val))
        
        pred = fcnn_model.prediction(encoded_test_imgs)
        print("Confusion Matrix of Test Data")
        fcnn_model.confusion_matrix(pred, y_test)
        fcnn_test_acc.append(fcnn_model.accuracy(pred, y_test))

        

    df = pd.DataFrame({
        "FCNN Architecture": fcnn_arch,
        "Training Accuracy": fcnn_train_acc,
        "Test Accuracy": fcnn_test_acc,
        "Validation Accuracy": fcnn_val_acc
    })
    print(df)


# In[131]:


fcnn_on_compressed()


# In[132]:


weights = autoencoder_noise.layers[2].weights[0].shape
print(weights)


# In[135]:


def weight_visualization(best_arch):

    weights = best_arch.layers[2].get_weights()[0]
    print("Weights Shape:",weights.shape)
    weights = weights.T
    i = 0
    k = 0
    
    # t = 0
    for i in range(16):
        plt.figure(figsize=(20, 4))
        j = 4*i
        for k in range(4):
            plt.subplot(1,4,k+1)
            plt.imshow(weights[j+k].reshape((28,28)))
            plt.title("Neuron {}".format(j+k+1))
            # t+=1
        plt.show()


# In[136]:


weight_visualization(autoencoder_noise)


# In[146]:



bottle_neck_dim = 128

autoencoder_noise = tf.keras.Sequential([
     tf.keras.Input((28, 28)),
     tf.keras.layers.Flatten(),
     Noise(stddev = 0.01, p=0.4),
     tf.keras.layers.Dense(bottle_neck_dim, activation = "sigmoid"),
     tf.keras.layers.Dense(784, activation = "sigmoid"),
    tf.keras.layers.Reshape((28, 28))
    ])


# In[147]:


input_img = tf.keras.layers.Input(shape=(28,28))
noisy_input = tf.keras.Model(input_img, autoencoder_noise.layers[1](input_img))

# Encoder 
encoder = tf.keras.models.Sequential(
    [
     input_img,
     tf.keras.layers.Flatten(),
     autoencoder_noise.layers[1], 
     autoencoder_noise.layers[2]
    ]
)
encoder.summary()

# Decoder
encoding_dim = tf.keras.Input((128, ))
decoder = tf.keras.models.Sequential([
    encoding_dim,
    autoencoder_noise.layers[-2],
    autoencoder_noise.layers[-1]
])

decoder.summary()


# In[148]:


myCallBack = ModelCallBack()
autoencoder_noise.compile(tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,beta_2=0.999,epsilon=1e-08),
              loss="mean_squared_error", metrics = ["mean_squared_error"])
autoencoder_noise.fit(train_data, train_data, epochs=1000, batch_size=32, verbose=0, callbacks=[myCallBack])


# In[149]:


def plot_AE_Re_and_orig_images(best_arch, data, randomlist):
     
#     randomlist = random.sample(range(len(data)), n)
    n = len(randomlist)
    plt.figure(figsize=(20, 4))
    t = 0
    for i in randomlist:
        encoded_imgs = encoder(train_data[i].reshape((1, 28, 28))).numpy()
        decoded_imgs = decoder(encoded_imgs).numpy()
        # display original
        ax = plt.subplot(2, n, t + 1)
        plt.imshow(train_data[i])
        plt.title("original")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, t + 1 + n)
        plt.imshow(decoded_imgs.reshape((28,28)))
        plt.title("reconstructed")
#         plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        t += 1
    plt.show()
#     return randomlist


# In[150]:


plot_AE_Re_and_orig_images(autoencoder_noise, train_data, h1_rl_train)


# In[151]:


plot_AE_Re_and_orig_images(autoencoder_noise, val_data, h1_rl_val)


# In[152]:



from fcnn import fcnn

def fcnn_on_compressed():

    encoded_train_imgs =encoder(train_data).numpy()
    encoded_test_imgs = encoder(test_data).numpy()
    encoded_val_imgs = encoder(val_data).numpy()

    fcnn_arch = [[64], [64,32,16],[128,32], [64,32], [32]]
#     best_arch_layers = best_arch.hidden_layers
#     input = best_arch_layers[len(best_arch_layers)//2]
#     print("fcnn Input:", input)
    fcnn_train_acc = []
    fcnn_test_acc = []
    fcnn_val_acc = []
    for arch in fcnn_arch:
        print("-"*25+"FCNN with Architecture: "+str(arch)+"-"*25)
        fcnn_model = fcnn((bottle_neck_dim,), arch, 5, "adam" ,False)
        fcnn_model._fit_(np.array(encoded_train_imgs), np.array(y_train))
        pred = fcnn_model.prediction(encoded_train_imgs)
        print("Confusion Matrix of Training Data")
        fcnn_model.confusion_matrix(pred, y_train)
        fcnn_train_acc.append(fcnn_model.accuracy(pred, y_train))

        pred = fcnn_model.prediction(encoded_val_imgs)
        print("Confusion Matrix of Validation Data")
        fcnn_model.confusion_matrix(pred, y_val)
        fcnn_val_acc.append(fcnn_model.accuracy(pred, y_val))
        
        pred = fcnn_model.prediction(encoded_test_imgs)
        print("Confusion Matrix of Test Data")
        fcnn_model.confusion_matrix(pred, y_test)
        fcnn_test_acc.append(fcnn_model.accuracy(pred, y_test))

        

    df = pd.DataFrame({
        "FCNN Architecture": fcnn_arch,
        "Training Accuracy": fcnn_train_acc,
        "Test Accuracy": fcnn_test_acc,
        "Validation Accuracy": fcnn_val_acc
    })
    print(df)


# In[153]:


fcnn_on_compressed()


# In[154]:


weight_visualization(autoencoder_noise)


# In[ ]:





# In[ ]:




