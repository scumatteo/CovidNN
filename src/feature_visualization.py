import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from tensorflow import keras
from keras.models import Model

def visualize(image, model, layer_name):
    #target_size = (model.input.shape[1], model.input.shape[2])
    #feature_maps = __extract_features(model, layer_name, __get_img_array(img_path=img_path, size = target_size))
    feature_maps = __extract_features(model, layer_name, image)
    __vis_feature_map(feature_maps)

# extracts the features of an image in a specific layer of a specific model.
def __extract_features(model, layer_name, image):
    feature_model = Model(inputs=model.inputs,outputs=model.get_layer(layer_name).output)
    return feature_model.predict(image)

# utility function to display 8 x 8 feature extracted (or less if there isn't).
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