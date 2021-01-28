import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model
import matplotlib.pyplot as plt
import os
import random


#funzione che restituisce un immagine nel formato corretto per la prediction.
def get_img_array(img_path, size = (200, 200)):
    image = keras.preprocessing.image.load_img(img_path, target_size=size)
    image = keras.preprocessing.image.img_to_array(image)
    # trasformazione in range [0, 1]
    image = image / 255
    #Aggiunta di una dimensione per convertire la nostra img in un "batch"
    image = np.expand_dims(image, axis=0)
    return image

#funzione che restituisce una immagine casuale all'interno di una cartella.
def get_random_image_by_path(path):
  return [path + name for name in os.listdir(path)][random.randint(0, len(os.listdir(path)))]


#funzione per la visualizzazione delle immagini in un array e dei rispettivi titoli.
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