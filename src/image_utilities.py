import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model

# returns an image in the right format to do a prediction.
def __get_img_array(img_path, size):
    # image is a PIL image of size 299x299
    image = keras.preprocessing.image.load_img(img_path, target_size=size)
    # image is a float32 Numpy array of shape (299, 299, 3)
    image = keras.preprocessing.image.img_to_array(image)
    # transform in range [0, 1]
    image = image / 255
    # We add a dimension to transform our image into a "batch"
    # of size (1, 299, 299, 3)
    image = np.expand_dims(image, axis=0)
    return image
