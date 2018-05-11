import os

from util.colorspace.colorspaceclasses import ColorMappingInitializer, ColorFrequencyCalculator
from util.colorspace.mapping import get_mapper


def get_mapping_with_class_weights(classifier, color_space, image_generator, image_size, nb_batches, scale_factor,
                                   weights_file_path=None, calculate_weights=True):
    class_weights = None
    if classifier:
        mapping = ColorMappingInitializer(scale_factor=scale_factor)
        mapping.initialize()
        data_mapper = get_mapper(color_space=color_space, classifier=classifier,
                                 color_to_class=mapping.color_to_class, class_to_color=mapping.class_to_color,
                                 factor=mapping.scale_factor)
        class_weight_calculator = ColorFrequencyCalculator(color_to_class=mapping.color_to_class,
                                                           class_to_color=mapping.class_to_color,
                                                           rgb_image_to_classes=data_mapper.rgb_to_classes,
                                                           image_generator=image_generator,
                                                           image_size=image_size)
        if not calculate_weights:
            pass
        elif weights_file_path:
            if os.path.exists(weights_file_path):
                class_weights = class_weight_calculator.load_weights(weights_file_path)
            else:
                class_weight_calculator.populate(num_batches=nb_batches)
                class_weights = class_weight_calculator.get_class_weights()
                class_weight_calculator.save_weights(weights_file_path)
        else:
            class_weight_calculator.populate(num_batches=nb_batches)
            class_weights = class_weight_calculator.get_class_weights()
    else:
        data_mapper = get_mapper(color_space=color_space, classifier=classifier)

    return data_mapper, class_weights
