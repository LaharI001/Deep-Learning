import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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


class fcnn:
    
    def __init__(self, input_shape,hidden_layers, output_layer, optimizer, isImage):
        self.layers = hidden_layers
        self.output_layer = output_layer
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.model = tf.keras.models.Sequential()
        if isImage:
            self.model.add(keras.layers.Flatten(input_shape=input_shape))
        else:
            self.model.add(tf.keras.layers.Input(shape = self.input_shape))
        for i in hidden_layers:
            self.model.add(tf.keras.layers.Dense(i, activation="relu"))
        self.model.add(tf.keras.layers.Dense(output_layer, activation = "softmax"))        

        self.model.summary()
        self.actual_classes = [0,1,2,3,4]
        self.optimizer = optimizer        
    
    def _fit_(self, X_train, Y_train, epochs=1000, batch_size = 32):
     
        Y_train = mapping_labels(Y_train, True)
      
        self.model.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy             
              optimizer=self.optimizer,
              metrics=["accuracy", "sparse_categorical_crossentropy"])
        
        self.model.fit(X_train,
          Y_train,
          epochs = epochs,
          batch_size = batch_size,
          verbose=0,
          shuffle=True,
          callbacks = [model_callback]
        )
        # print(model_callback.end_epoch)
        self.end_epoch = model_callback.end_epoch
        self.avg_error = model_callback.avg_error

    def plot_err_vs_epoch(self, title):
        print(len(self.avg_error), len(range(self.end_epoch+1)))
        plt.plot( range(self.end_epoch+1), self.avg_error)
        # print(self.layer1_nodes, self.layer2_nodes, self.layer3_nodes)
        plt.xlabel("No. of Epochs")
        plt.ylabel("Average Errors")
        title = "Average Error Vs Epoch \nArchitecture : "
        for i in self.layers:
            title+=(str(i)+"")
        # {}, {}, {}, {}, {}]".format(title, 784,self.layer[0], self.layer2_nodes, self.layer3_nodes, 5)
        plt.title(title)
        plt.show()


    def accuracy(self, predict, actual):
        return accuracy_score(predict, actual)

    def confusion_matrix(self, predict, actual):
        cm= confusion_matrix(actual, predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.actual_classes)
        disp.plot()
        plt.show()
        # return cm

    def prediction(self, X_test):
        pred = self.model.predict(X_test)
        pred = [np.argmax(i) for i in pred]
        return mapping_labels(pred, False)

        