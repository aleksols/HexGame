from tensorflow.keras.layers import Dense

general = {
    "size": 4,
    "starting player": 1,
    "simulations": 50,
    "actual games": 200,
    "num anets": 5,
    "tournament games": 25
}


network = {
    "layers": [Dense, Dense],
    "dimensions": [64, general["size"] ** 2],
    "activations": ["relu", "softmax"],
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "batch size": 32
}