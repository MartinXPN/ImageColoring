import random
import numpy as np
import threading

class BaseBatchGenerator:
    
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
            batch_features = []
            batch_labels   = []

            for i in range(self.batch_size):
                path = self.image_paths[self.index]
                x, y = generate_one(path)

                self.index += 1
                if self.index >= len(self.image_paths):
                    self.index = 0
            
            return np.array(batch_features), np.array(batch_labels)
    
    def generate_one(path):
        raise NotImplementedError("Please Implement this method")


class DiscriminatorBatchGenerator(BaseBatchGenerator):
    """"
    Generates real images to make the discriminator better in predicting real images
    """"
    
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(DiscriminatorBatchGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)
        
    def generate_one(path):
        image_gray = load_img(path, grayscale=True, target_size=(self.image_height, self.image_width))
        x = np.zeros((self.image_height, self.image_width, 3))
        x[:,:,0] = image_gray - greyscale_image_mean
        x[:,:,1] = image_gray - greyscale_image_mean
        x[:,:,2] = image_gray - greyscale_image_mean
        y = True
        
        return x, y


class ColorizerBatchGenerator(BaseBatchGenerator):
    """
    Generates input for colorizer to improve it's performance in fooling the discriminator
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(DiscriminatorBatchGenerator, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

        
    def generate_one(path):
        image = io.imread(path)
        image = resize(image, output_shape=(self.image_height, self.image_width, 3))
        image_gray = load_img(path, grayscale=True, target_size=(self.image_height, self.image_width))
        x = color.rgb2lab(image)
        x[:,:,0] = image_gray - greyscale_image_mean
        y = get_label()
        
        return x, y
   
    def get_label():
        return True
    
class DiscriminatorBatchGeneratorForColorizer(ColorizerBatchGenerator):
    """
    Generates input for colorizer to improve discriminator's performance in detecting fake images
    """
    def __init__(self, image_paths, batch_size, image_height, image_width, shuffle = True):
        super(DiscriminatorBatchGeneratorForColorizer, self).__init__(image_paths, batch_size, image_height, image_width, shuffle)

    def get_label():
        return False