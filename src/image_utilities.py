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
    n_rows = len(images)//n_columns
    n_cols = n_columns
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15,15))
    if n_rows == 1:
      for i in range(n_cols):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i])
    else:
      for i in range(n_rows):
        for j in range(n_cols):
          pos = j + i*n_cols
          ax[i, j].imshow(images[pos])
          ax[i, j].set_title(titles[pos])
          plt.tight_layout()
          plt.show()