

general = {
    "size": 4,
    "starting player": 1,
    "simulations": 50,
    "actual games": 200,
    "num anets": 5,
    "tournament games": 25
}


network = {
    "dimensions": [general["size"] ** 2 + 1, 64, 64, general["size"] ** 2],
    "activations": ["relu", "relu", "softmax"],
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "batch size": 32
}