import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model

# returns an image in the right format to do a prediction.
def get_img_array(img_path, size):
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


def vis_images(normal_img, covid_img, pneum_img):
    fig, ax = plt.subplots(1, 3, figsize=(15,15))
    ax[0].imshow(normal_img)
    ax[0].set_title("Normal")
    ax[1].imshow(covid_img)
    ax[1].set_title("Covid-19")
    ax[2].imshow(pneum_img)
    ax[2].set_title("Viral Pneumonia")
    plt.tight_layout()
    plt.show()