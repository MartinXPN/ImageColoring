def make_trainable(layers, is_trainable):
    for l in layers:
        l.trainable = is_trainable
