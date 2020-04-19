game = {
    "size": 3,
    "starting player": 1
}

tournament = {
    "games": 25,
    "agent policies": "probabilistic",  # greedy, probabilistic or random
    "file prefix": f"{game['size']}",
    "visualize": [1, 2, 3]
}

network = {
    "dimensions": [2 * game["size"] ** 2 + 2] + [64, 64] + [game["size"] ** 2],
    "activations": ["relu"] + ["relu", "relu"] + ["softmax"],
    "output args": {"dim": 1},
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "learning rate": 0.006
}

training = {
    "batch size": 32,
    "epochs": 50,
    "simulations": 50,
    "actual games": 2000,
    "num anets": 5,
    "file prefix": f"{game['size']}",
    "visualize": [],  # What games to visualize. F.eks [1, 50, 200]. 1 indexed
    "buffer max size": 2000
}
