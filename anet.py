import tensorflow as tf
from config import network


def ANET() -> tf.keras.models.Sequential:
    model = tf.keras.Sequential()
    for layer, dim, activation in zip(network["layers"], network["dimensions"], network["activations"]):
        model.add(layer(dim, activation=activation))
    return model.compile(optimizer=network["optimizer"], loss=network["loss"])
