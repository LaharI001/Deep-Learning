import numpy as np
import matplotlib.pyplot as plt

class Convolution:

    def __init__(self, filters):
        self.filters = filters
        self.fr, self.fc, self.fd, self.num_filter = filters.shape 
    def iterate_regions( self, image):
        
        h = image.shape[0]
        w = image.shape[1]


        for i in range(h - self.fr + 1):
            for j in range(w - self.fc + 1):
                im_region = image[i:(i + self.fr), j:(j + self.fc)]
                yield im_region, i, j

    def generate_feature_map(self, image):
        self.ir, self.ic, self.id = image.shape
        self.output = np.zeros((self.ir - self.fr + 1, self.ic - self.fc + 1,self.num_filter))
        
        for im_region, i, j in self.iterate_regions(image):
            self.output[i, j] = np.sum(im_region * np.transpose(self.filters,(3,0,1,2)), axis=( 1,2,3))
    
        return self.output
    def relu(self, data):
        data[data<=0] = 0
        return data

    def plot(self, temp_list):
        for i in temp_list:
            print("Filter: ", i+1)
            for channel in range(3):
                plt.subplot(1,3,channel+1)
                plt.imshow(self.filters[:, :, channel, i])
                plt.axis("off")
                plt.title("Channel: "+str(channel+1), fontsize=8)
            plt.show()
        
        i=0
        while i<10:
            j=0
            high=min(4, 10-i)
            for j in range(high):
                plt.subplot(1,4,j+1)
                plt.imshow(self.output[:, :, temp_list[i]])
                plt.title("Filter: "+str(temp_list[i]+1))
                plt.axis("off")
                i+=1
            plt.show()

