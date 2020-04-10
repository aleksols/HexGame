import tensorflow as tf
from config import network
import config


def ANET() -> tf.keras.models.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(config.general["size"] ** 2 + 1)))
    for layer, dim, activation in zip(network["layers"], network["dimensions"], network["activations"]):
        model.add(layer(dim, activation=activation))
    model.compile(optimizer=network["optimizer"], loss=network["loss"])
    return model

