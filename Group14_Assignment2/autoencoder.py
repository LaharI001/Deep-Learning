import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Model

class ModelCallBack(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.avg_error = []
        self.prev_error = 0
        self.current_error = 0
        self.end_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        if epoch == 0:
            self.prev_error = logs["sparse_categorical_crossentropy"]
            self.current_error = logs["sparse_categorical_crossentropy"]
        else:
            self.current_error = logs["sparse_categorical_crossentropy"]
            if abs(self.current_error-self.prev_error)<1e-4:
                # self.end_epoch = epoch
                self.model.stop_training = True
            else:
                self.prev_error = self.current_error
        # print(logs["sparse_categorical_crossentropy"])
        print("Epoch", epoch, "done | loss:", logs["loss"])
        self.end_epoch = epoch
        self.avg_error.append(self.prev_error)

model_callback = ModelCallBack()

def mapping_labels(labels, bool_val):
    d = {0:0,2:1, 3:2, 7:3, 8:4}
    s = {0:0, 1:2, 2:3, 3:7, 4:8}
    if bool_val:
        return np.array([d[labels[i]] for i in range(len(labels))])
    else:
        return np.array([s[labels[i]] for i in range(len(labels))])

class Autoencoder(Model):
    def __init__(self, hidden_layers):
        super(Autoencoder, self).__init__()
        # self.bottleneck_dim = bottleneck_dim
        total_hidden_layers = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.encoder = tf.keras.Sequential()
        self.encoder.add(keras.layers.Flatten())

        self.decoder = tf.keras.Sequential()
        for i in range(total_hidden_layers):  
            if i < ((total_hidden_layers//2)+1):
                self.encoder.add(keras.layers.Dense(hidden_layers[i], activation='sigmoid'))
            else:
                self.decoder.add(keras.layers.Dense(hidden_layers[i], activation='sigmoid'))
        
        self.decoder.add(keras.layers.Dense(784, activation='sigmoid'))
        self.decoder.add(keras.layers.Reshape((28, 28)))
        
        # self.encoder.summary()
        # self.decoder.summary()
        self.avg_error = []
        self.end_epoch = 0
        
        
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


