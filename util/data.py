import numpy as np
from keras.utils import GeneratorEnqueuer


class ImageGenerator(object):
    def __init__(self, rgb_generator,
                 input_processing_function=lambda x: x,
                 label_processing_function=lambda x: None,
                 use_multiprocessing=False, wait_time=0.01,
                 workers=1, max_queue_size=10):
        class BatchGenerator(object):
            def __next__(self):
                rgb_images = next(rgb_generator)
                inputs, labels = [], []
                if type(rgb_images) is tuple:
                    rgb_images, labels = rgb_images

                inputs = np.array([input_processing_function(rgb_image) for rgb_image in rgb_images])
                if len(labels) == 0:
                    labels = [label_processing_function(rgb_image) for rgb_image in rgb_images]
                    labels = [item for item in labels if item is not None]

                if len(labels) == 0:    return np.array(inputs)
                else:                   return np.array(inputs), np.array(labels)

        self.generator = GeneratorEnqueuer(generator=BatchGenerator(),
                                           use_multiprocessing=use_multiprocessing,
                                           wait_time=wait_time)
        self.generator.start(workers=workers, max_queue_size=max_queue_size)

    def __next__(self):
        return next(self.generator.get())
