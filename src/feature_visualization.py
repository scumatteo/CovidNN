import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model

#funzione pubblica per visualizzare le feature map calcolate.
def visualize(image, model, layer_name):
    feature_maps = __extract_features(model, layer_name, image)
    __vis_feature_map(feature_maps)

# estrazione delle features di un immagine in uno specifico livello di uno specifico modello.
def __extract_features(model, layer_name, image):
    feature_model = Model(inputs=model.inputs,outputs=model.get_layer(layer_name).output)
    return feature_model.predict(image)

#funzione di utilità per visualizzare 8 x 8 feature estratte (o meno se non sono presenti).
def __vis_feature_map(feature_maps):
    ncol = min(8,int(np.floor(np.sqrt(feature_maps.shape[3]))))
    fig, ax = plt.subplots(ncol, ncol,figsize=(2*ncol,ncol*1.5))
    if ncol == 1:
        ax.imshow(feature_maps[0,:,:,0],cmap="gray")
    else:
        count = 0
        for i in range(ncol):
            for j in range(ncol):
                ax[j,i].imshow(feature_maps[0,:,:,count],cmap="gray")
                ax[j,i].axis("off")
                count += 1
    plt.tight_layout()
    plt.show()