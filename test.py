import matplotlib.pyplot as plt
import random
import numpy as np


fig, ax = plt.subplots(1, figsize=(15, 7))
x = np.arange(0, 100)

line, = ax.plot(x)
print(line)
plt.show(block=False)
while True:
    # x.append(random.randint(1, 4))
    x = np.random.randn()
    line.set_ydata(x)
    fig.canvas.draw()
    fig.canvas.flush_events()
