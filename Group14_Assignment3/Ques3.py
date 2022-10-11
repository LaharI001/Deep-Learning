import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from cnn import CNN
from convolution import Convolution



def convert(image):
    im=np.transpose(image,(2,0,1))
    return im


def load_images_from_folder(folder):
    images = []
    y = []
#     d = {0:0,2:1, 3:2, 7:3, 8:4}
    for i in folder.keys():
        for filename in os.listdir(i):
#             print(filename)
            img = cv2.imread(os.path.join(i,filename))
            img = np.array(img)
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            img = convert(img)
            # print(img.shape)
            img = img/255
            if img is not None:
                images.append(np.asarray(img))
            # print(int(i[-1]), end="")
            y.append(folder[i])
    images = np.array(images)
    y = np.array(y)
    x_data , y_data = shuffle (images, y)
    print(x_data.shape, y.shape)
    return x_data,y_data



cnn = CNN()
train_data,Y_train = load_images_from_folder({"./Group_14/train/grand_piano":0,"./Group_14/train/starfish":1,"./Group_14/train/trilobite":2})

cost = cnn.train(train_data, Y_train, num_epochs=1)
params, cost = pickle.load(open("params.pkl", 'rb'))
[f1, f2, w3, w4, b1, b2, b3, b4] = params

test_data,y_test = load_images_from_folder({"./Group_14/test/grand_piano":0,"./Group_14/test/starfish":1,"./Group_14/test/trilobite":2})
val_data, y_val = load_images_from_folder({"./Group_14/val/grand_piano":0,"./Group_14/val/starfish":1,"./Group_14/val/trilobite":2})

train_pred = cnn.predict(train_data)
test_pred = cnn.predict(test_data)
val_pred = cnn.predict(val_data)


print("Accuracy Score of Training Data:", cnn.accuracy(train_pred, Y_train))
print("Confusion Matrix of Training Data: ")
cnn.conf_matrix(train_pred, Y_train)

print("Accuracy Score of Validation Data:", cnn.accuracy(val_pred, y_val))
print("Confusion Matrix of Validation Data: ")
cnn.conf_matrix(val_pred, y_val)

print("Accuracy Score of Test Data:", cnn.accuracy(test_pred, y_test))
print("Confusion Matrix of Test Data: ")
cnn.conf_matrix(test_pred, y_test)



params1 = pickle.load(open("kaiming_params_update.pkl", 'rb'))
params2 = pickle.load(open("kaiming_params.pkl", 'rb'))
def plot_10filters_and_feature(image):
    print("-"*25+"Convolve Layer 1"+"-"*25)
    layer1= Convolution(params1[0])
    f_map1 = layer1.generate_feature_map(image)
    f_map1 = layer1.relu(f_map1)
#     print(params)
    layer1.plot(params2[2])
    # convLayer1.append(_)
#     print(f_map1.shape)
    print("-"*25+"Convolve Layer 2"+"-"*25)
    layer2= Convolution(params1[1])
    f_map2 = layer2.generate_feature_map(f_map1)
    f_map2 = layer1.relu(f_map2)
    layer2.plot(params2[3])

def read_image(path):
    img1 = cv2.imread(path)
    img = np.array(img1)
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
    return img

img1 = read_image("./Group_14/train/grand_piano/image_0001.jpg")
img2 = read_image("./Group_14/train/starfish/image_0002.jpg")
img3 = read_image("./Group_14/train/trilobite/image_0001.jpg")

print("Image 1: Grand Piano")
plot_10filters_and_feature(img1)
print("Image 2: StarFish")
plot_10filters_and_feature(img2)
print("Trilobite")
plot_10filters_and_feature(img3)


