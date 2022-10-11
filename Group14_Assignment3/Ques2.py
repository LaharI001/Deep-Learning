import pickle
from convolution import Convolution
import cv2
import numpy as np

params = pickle.load(open("kaiming_params.pkl", 'rb'))
def plot_10filters_and_feature(image):
    print("-"*25+"Convolve Layer 1"+"-"*25)
    layer1= Convolution(params[0])
    f_map1 = layer1.generate_feature_map(image)
    f_map1 = layer1.relu(f_map1)
#     print(params)
    layer1.plot(params[2])
    # convLayer1.append(_)
#     print(f_map1.shape)
    print("-"*25+"Convolve Layer 2"+"-"*25)
    layer2= Convolution(params[1])
    f_map2 = layer2.generate_feature_map(f_map1)
    f_map2 = layer1.relu(f_map2)
    layer2.plot(params[3])


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