import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model
import numpy as np

def visualize_feature_map(img_path, size, model, layer_name):
    feature_maps = extract_features(model, layer_name, get_img_array(img_path=img_path, size = size))
    vis_feature_map(feature_maps)

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

# extracts the features of an image in a specific layer of a specific model.
def extract_features(model, layer_name, image):
    feature_model = Model(inputs=model.inputs,outputs=model.get_layer(layer_name).output)
    return feature_model.predict(image)

# utility function to display 8 x 8 feature extracted (or less if there isn't).
def vis_feature_map(feature_maps):
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