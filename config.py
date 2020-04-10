from tensorflow.keras.layers import Dense

general = {
    "size": 4,
    "starting player": 1,
    "simulations": 500
}


network = {
    "layers": [Dense, Dense, Dense],
    "dimensions": [general["size"] ** 2 + 1, 64, 32, general["size"] ** 2],
    "activations": ["relu", "relu", "softmax"],
    "optimizer": "adam",
    "loss": "categorical_crossentropy"
}