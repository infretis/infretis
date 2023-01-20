import numpy as np
import matplotlib.pyplot as plt


cap = 253
a = np.loadtxt('./pattern.txt')[:cap]
min0 = min(a[:, 1])
a[:,1] -= min0
a[:,2] -= min0

workers = list(set(a[:, 3]))
min_ens, max_ens = min(a[:,0]), max(a[:,0])
c_dic = {0: 'r', 1: 'b', 2: 'g'}
dic = {int(worker): [] for worker in workers}
label_workers = []
for i in range(len(a)):
    worker = int(a[i][3])
    if worker not in label_workers:
        plt.plot([a[i][1], a[i][2]], [a[i][0]]*2, color=c_dic[worker], marker="|", label=f'worker {worker}')
        label_workers.append(worker)
    else: 
        plt.plot([a[i][1], a[i][2]], [a[i][0]]*2, color=c_dic[worker], marker="|")

for worker in workers:
    x = np.array([(i) for i in a if i[3] == int(worker)])
    for i in range(len(x)):
        if i + 1 < len(x) and x[i][0] != x[i+1][0]:
            dic[int(worker)].append([[x[i][2]]*2, [x[i][0], x[i+1][0]]])
    for i in dic[int(worker)]:
        plt.plot(i[0], i[1], alpha=0.2, color=c_dic[int(worker)])

plt.plot([3, max(a[:, 2])-3], [max_ens+4]*2, color='k') 
plt.plot([3, max(a[:, 2])-3], [min_ens-4]*2, color='k') 
plt.xlabel(r"Time [s]")
plt.ylabel(r"OP")
plt.legend(frameon=False, loc='upper right')
plt.savefig('pattern.pdf')
