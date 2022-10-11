
import numpy as np
import matplotlib.pyplot as plt
import pickle
# from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



class CNN:
    def nanargmax(self,arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs

    def convolution(self, image, filt, bias, s=1):
   
        (n_f, n_c_f, f, _) = filt.shape 
        n_c, in_dim, _ = image.shape
        
        out_dim = int((in_dim - f)/s)+1 
        assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
        out = np.zeros((n_f,out_dim,out_dim))
        
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            
        return out

    def maxpool(self, image, f=2, s=2):
        
        n_c, h_prev, w_prev = image.shape
        
        h = int((h_prev - f)/s)+1
        w = int((w_prev - f)/s)+1
        
        downsampled = np.zeros((n_c, h, w))
        for i in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return downsampled

    def softmax(self, X):
        out = np.exp(X)
        return out/np.sum(out)

    def categoricalCrossEntropy(self, probs, label):
        return -np.sum(label * np.log(probs))

            
    def convolutionBackward(self, dconv_prev, conv_in, filt, s):
        
        (n_f, n_c, f, _) = filt.shape
        (_, orig_dim, _) = conv_in.shape
        
        dout = np.zeros(conv_in.shape) 
        dfilt = np.zeros(filt.shape)
        dbias = np.zeros((n_f,1))
        for curr_f in range(n_f):
            
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                    dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            
            dbias[curr_f] = np.sum(dconv_prev[curr_f])
        
        return dout, dfilt, dbias



    def maxpoolBackward(self, dpool, orig, f, s):
        
        (n_c, orig_dim, _) = orig.shape
        
        dout = np.zeros(orig.shape)
        
        for curr_c in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    (a, b) = self.nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                    dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                    
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            
        return dout


    def conv(self, image, label, params, conv_s, pool_f, pool_s):
        
        [f1, f2, w3, w4, b1, b2, b3, b4] = params 
        
        conv1 = self.convolution(image, f1, b1, conv_s) 
        conv1[conv1<=0] = 0 
        
        conv2 = self.convolution(conv1, f2, b2, conv_s) 
        conv2[conv2<=0] = 0
        
        pooled = self.maxpool(conv2, pool_f, pool_s)
        
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1)) 
        
        z = w3.dot(fc) + b3 
        z[z<=0] = 0 
        
        out = w4.dot(z) + b4 
        
        probs = self.softmax(out) 
        
        
        loss = self.categoricalCrossEntropy(probs, label) 
            
    
        dout = probs - label
        dw4 = dout.dot(z.T) 
        db4 = np.sum(dout, axis = 1).reshape(b4.shape) 
        
        dz = w4.T.dot(dout) 
        dz[z<=0] = 0 
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis = 1).reshape(b3.shape)
        
        dfc = w3.T.dot(dz) 
        dpool = dfc.reshape(pooled.shape)
        
        dconv2 = self.maxpoolBackward(dpool, conv2, pool_f, pool_s) 
        dconv2[conv2<=0] = 0 
        
        dconv1, df2, db2 = self.convolutionBackward(dconv2, conv1, f2, conv_s)
        dconv1[conv1<=0] = 0
        
        dimage, df1, db1 = self.convolutionBackward(dconv1, image, f1, conv_s) 
        
        grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 
        
        return grads, loss



    def Gradient_Descent(self,batch,y, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
        
        [f1, f2, w3, w4, b1, b2, b3, b4] = params
        
        X = batch 
        Y = y 
        
        cost_ = 0
        batch_size = len(batch)
        
        df1 = np.zeros(f1.shape)
        df2 = np.zeros(f2.shape)
        dw3 = np.zeros(w3.shape)
        dw4 = np.zeros(w4.shape)
        db1 = np.zeros(b1.shape)
        db2 = np.zeros(b2.shape)
        db3 = np.zeros(b3.shape)
        db4 = np.zeros(b4.shape)
        
        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)
        
        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)
        
        for i in range(batch_size):
            
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) 
            
            grads, loss = self.conv(x, y, params, 1, 2, 2)
            [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
            
            df1+=df1_
            db1+=db1_
            df2+=df2_
            db2+=db2_
            dw3+=dw3_
            db3+=db3_
            dw4+=dw4_
            db4+=db4_

            cost_+= loss

        # Parameter Update  
            
        v1 = beta1*v1 + (1-beta1)*df1/batch_size 
        s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2
        f1 -= lr * v1/np.sqrt(s1+1e-7) 
        
        bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
        bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
        b1 -= lr * bv1/np.sqrt(bs1+1e-7)
    
        v2 = beta1*v2 + (1-beta1)*df2/batch_size
        s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
        f2 -= lr * v2/np.sqrt(s2+1e-7)
                        
        bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
        bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
        b2 -= lr * bv2/np.sqrt(bs2+1e-7)
        
        v3 = beta1*v3 + (1-beta1) * dw3/batch_size
        s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
        w3 -= lr * v3/np.sqrt(s3+1e-7)
        
        bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
        bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
        b3 -= lr * bv3/np.sqrt(bs3+1e-7)
        
        v4 = beta1*v4 + (1-beta1) * dw4/batch_size
        s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
        w4 -= lr * v4 / np.sqrt(s4+1e-7)
        
        bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
        bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
        b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
        

        cost_ = cost_/batch_size
        cost.append(cost_)

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
        
        return params, cost


    def train(self,train_data, Y, num_classes = 3, lr = 0.001, beta1 = 0.95, beta2 = 0.99, img_dim = 224, img_depth = 3, f = 3, num_filt1 = 32, num_filt2 = 64, batch_size = 32, num_epochs = 20, save_path = 'params.pkl'):

        
        f1, f2, w3, w4 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), (128,774400), (3, 128)
        f1 = self.initializeFilter(f1)
        f2 = self.initializeFilter(f2)
        w3 = self.initializeWeight(w3)
        w4 = self.initializeWeight(w4)

        b1 = np.zeros((f1.shape[0],1))
        b2 = np.zeros((f2.shape[0],1))
        b3 = np.zeros((w3.shape[0],1))
        b4 = np.zeros((w4.shape[0],1))

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
        with open("kaiming.pkl", 'wb') as file:
            pickle.dump(params, file)
        cost = []

        print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

        for epoch in range(num_epochs):
            
            batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]
            y_batches = [Y[k:k + batch_size] for k in range(0, Y.shape[0], batch_size)]
           
            i = 0
            print("Epoch {} start".format(epoch+1))
            for y,batch in zip(y_batches, batches):
                params, cost = self.Gradient_Descent(batch,y, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
                
                print("Batch {} : Loss {}".format(i+1, cost[-1]))
                i+=1
            print("Epoch {} done".format(epoch+1))    
        to_save = [params, cost]
        
        with open("params.pkl", 'wb') as file:
            pickle.dump(to_save, file)
            
        return cost

    def predict(self, image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 1, pool_f = 2, pool_s = 2):
        '''
        Make predictions with trained filters/weights. 
        '''
        conv1 = self.convolution(image, f1, b1, conv_s) 
        conv1[conv1<=0] = 0
        
        conv2 = self.convolution(conv1, f2, b2, conv_s)
        conv2[conv2<=0] = 0 
        
        pooled = self.maxpool(conv2, pool_f, pool_s) 
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1)) 
        
        z = w3.dot(fc) + b3 
        z[z<=0] = 0 
        
        out = w4.dot(z) + b4 
        probs = self.softmax(out) 
        
        return np.argmax(probs), np.max(probs)

    def initializeFilter(self,size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def initializeWeight(self,size):
        return np.random.standard_normal(size=size) * 0.01

    

    def accuracy( self,pred, actual):
        return accuracy_score(pred, actual)

    def conf_matrix(self, predict, actual):
        cm= confusion_matrix(actual, predict)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2])
        disp.plot()
        plt.show()   

