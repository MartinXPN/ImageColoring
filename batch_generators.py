import random
import numpy as np
import threading
from keras.preprocessing.image import load_img
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean


image_mean = (103.939, 116.779, 123.68)
greyscale_image_mean = np.mean(image_mean)


class BaseBatchGenerator(object):
    
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.lock = threading.Lock()
        self.index = 0
        if shuffle:
            random.shuffle(self.image_paths)

    def next(self):
        with self.lock:
            batch_inputs = []
            batch_labels   = []

            for i in range(self.batch_size):
                path = self.image_paths[self.index]
                x, y = self.generate_one(path)
                
                batch_inputs.append( x )
                batch_labels.append( y )

                self.index += 1
                if self.index >= len(self.image_paths):
                    self.index = 0
            
            return np.array(batch_inputs), np.array(batch_labels)
    
    def generate_one(self, path):
        raise NotImplementedError("Please Implement this method")


class DiscriminatorRealGenerator(BaseBatchGenerator):
    '''
    Generates real images to make the discriminator better in predicting real images
    '''
    
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(DiscriminatorRealGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)
        
    def generate_one(self, path):
        image = io.imread(path)
        image = resize(image, output_shape=(self.image_height, self.image_width, 3))
        image_gray = load_img(path, grayscale=True, target_size=(self.image_height, self.image_width))
        x = color.rgb2lab(image)
        x /= 128.                                     # scale a,b space to (-1;1)
        x[:,:,0] = image_gray - greyscale_image_mean  # first channel is simply the grayscale verison of image
        y = True
        
        return x, y


class ColorizerBatchGenerator(BaseBatchGenerator):
    """
    Generates input for colorizer to improve it's performance in fooling the discriminator
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(ColorizerBatchGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

        
    def generate_one(self, path):
        image_gray = load_img(path, grayscale=True, target_size=(self.image_height, self.image_width))
        x = np.zeros((self.image_height, self.image_width, 3))
        x[:,:,0] = image_gray - greyscale_image_mean
        x[:,:,1] = image_gray - greyscale_image_mean
        x[:,:,2] = image_gray - greyscale_image_mean
        y = self.get_label()
        
        return x, y
   
    def get_label(self):
        return True
    
class DiscriminatorFakeGenerator(ColorizerBatchGenerator):
    """
    Generates input for colorizer to improve discriminator's performance in detecting fake images
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(DiscriminatorFakeGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

    def get_label(self):
        return False