import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt


# returns an image in the right format to do a prediction.
def get_img_array(img_path, size):
    image = keras.preprocessing.image.load_img(img_path, target_size=size)
    image = keras.preprocessing.image.img_to_array(image)
    # transform in range [0, 1]
    image = image / 255
    # We add a dimension to transform our image into a "batch"
    image = np.expand_dims(image, axis=0)
    return image


def vis_images(images, titles, n_columns):
    fig, ax = plt.subplots(len(images)//n_columns, n_columns, figsize=(10,10))
    for i in len(images):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()