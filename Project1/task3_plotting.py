import numpy as np
import matplotlib.pyplot as plt

k=10
N=1000

xpoints = []
ypoints = []
for D in range(1,50):
    xpoints.append(D)
    ypoints.append( (k/N)**(1/D) )

plt.plot(xpoints, ypoints)
plt.xlabel("Dimensionality - D")
plt.ylabel("Edge - l")
plt.show()
