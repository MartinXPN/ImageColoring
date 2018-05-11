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
                inputs = input_processing_function(rgb_images)
                labels = label_processing_function(rgb_images)
                return inputs if labels is None else inputs, labels

        self.generator = GeneratorEnqueuer(generator=BatchGenerator(),
                                           use_multiprocessing=use_multiprocessing,
                                           wait_time=wait_time)
        self.generator.start(workers=workers, max_queue_size=max_queue_size)

    def __next__(self):
        return next(self.generator.get())
