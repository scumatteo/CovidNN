import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Model

def visualize(model, layer_name, image):
    target_size = (model.input.shape[1], model.input.shape[2])
    prediction = model.predict(image)
    classIdx = np.argmax(prediction[0])
    cam, cam3 = __compute_heatmap(model, layer_name, image, classIdx=classIdx, upsample_size=target_size)
    heatmap = __overlay_gradCAM(image,cam3)
    heatmap = heatmap[..., ::-1] # BGR to RGB
    __vis_heatmap(cam, cam3, heatmap)

def __compute_heatmap(model, layer_name, image, upsample_size, classIdx=None, eps=1e-5):
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        # record operations for automatic differentiation
            
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (conv_outs, preds) = grad_model(inputs)  # preds after softmax
            if classIdx is None:
                classIdx = np.argmax(preds)
            loss = preds[:, classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, conv_outs)
        # discard batch
        conv_outs = conv_outs[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outs), axis=-1)

        # Apply reLU
        camR = np.maximum(cam, 0)
        camR = camR / np.max(camR)
        camR = cv2.resize(camR, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(camR, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        
        return cam, cam3


def __overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.4 * cam3 / 255 + img

    return new_img

def __vis_heatmap(cam, cam3, heatmap):
    fig, ax = plt.subplots(1, 3, figsize=(50,50))
    ax[0].imshow(cam)
    ax[1].imshow(cam3)
    ax[2].imshow(heatmap)
    plt.tight_layout()
    plt.show()