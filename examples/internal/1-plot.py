import numpy as np
import matplotlib.pyplot as plt

# 002 1 worker
print(dir(np.loadtxt))
a = np.loadtxt('out.txt', skiprows=400)

plt.plot(a[:, 0], a[:, 1], label='misc')
plt.plot(a[:, 0], a[:, 2], label='loop')
plt.plot(a[:, 0], a[:, 3], label='random assign')
plt.plot(a[:, 0], a[:, 4], label='archive')
plt.plot(a[:, 0], a[:, 5], label='archive')
#plt.plot(*zip(*enumerate(a)), a)
#plt.scatter(b[:,0], b[:, 1], facecolors='none', edgecolors='r', label='scatter')

plt.xlabel(r"Time [ps]")
plt.ylabel(r"OP")
plt.legend(frameon=False)
plt.show()

