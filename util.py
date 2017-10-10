def make_trainable(layers, is_trainable):
    for l in layers:
        l.trainable = is_trainable


def last(arr, k=100):
    if arr:  return arr[-1]
    else:    return k
