game = {
    "size": 5,
    "starting player": 1
}

tournament = {
    "games": 100,
    "agent policies": "probabilistic",  # greedy, probabilistic or random
    "file prefix": f"{game['size']}",
    "visualize": []
}

network = {
    "h_dims": [64, 64],
    "activations": ["relu"] * 2,
    "output func": "softmax",
    "output args": {"dim": 1},
    "optim": "adam",
    "loss": "categorical_crossentropy",
    "learning rate": 0.01,
    "use conv": False
}

training = {
    "batch size": 32,
    "epochs": 1,
    "simulations": 500,
    "actual games": 200,
    "num anets": 5,
    "file prefix": f"{game['size']}",
    "visualize": [],  # What games to visualize. F.eks [1, 50, 200]. 1 indexed
    "buffer max size": 2000
}

conv_net = {
    "h_channels": [64],
    "activations": ["relu"],
    "output func": "softmax",
    "output args": {"dim": 1},
    "optim": "adam",
    "loss": "categorical_crossentropy",
    "learning rate": 0.0001,
}

# worked pretty well for greedy agents and size 3
# network = {
#     "h_dims": [64, 64],
#     "activations": ["relu"] * 2,
#     "output func": "softmax",
#     "output args": {"dim": 1},
#     "optim": "adam",
#     "loss": "categorical_crossentropy",
#     "learning rate": 0.001,
#     "use conv": False
# }
#
# training = {
#     "batch size": 32,
#     "epochs": 1,
#     "simulations": 500,
#     "actual games": 200,
#     "num anets": 5,
#     "file prefix": f"test_adam{game['size']}",
#     "visualize": [],  # What games to visualize. F.eks [1, 50, 200]. 1 indexed
#     "buffer max size": 2000
# }