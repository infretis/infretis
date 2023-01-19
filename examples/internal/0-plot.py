import numpy as np
import matplotlib.pyplot as plt

# 002 1 worker
a = np.loadtxt('data/infretis_002_pcross.txt')
b = np.loadtxt('data/pyretis_002_pcross.txt')

# whole 1 worker
c = np.loadtxt('data/infretis-total-probability.txt')
d = np.loadtxt('data/pyretis-total-probability.txt')
# whole 3 worker
e = np.loadtxt('data/infretis-total-probability_wfha3workers8ens.txt')
f = np.loadtxt('data/pyretis-total-probability_8ens.txt')

plt.plot(a[:, 0], a[:, 1], label='infretis')
plt.plot(b[:, 0], b[:, 1], label='pyretis')
plt.plot(c[:, 0], c[:, 1], label='infretis')
plt.plot(d[:, 0], d[:, 1], label='pyretis')
plt.plot(e[:, 0], e[:, 1], label='infretis-wfha3workers')
plt.plot(f[:, 0], f[:, 1], label='pyretis')
#plt.plot(*zip(*enumerate(a)), a)
#plt.scatter(b[:,0], b[:, 1], facecolors='none', edgecolors='r', label='scatter')

plt.xlabel(r"Time [ps]")
plt.ylabel(r"OP")
plt.legend(frameon=False)
plt.show()

