import numpy as np
import matplotlib.pyplot as plt

def pattern(inp, cap=250):
    a = np.loadtxt(inp)[:cap]
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
    
    plt.plot([3, max(a[:, 2])-3], [max_ens+3]*2, color='k', alpha=0.0)
    plt.plot([3, max(a[:, 2])-3], [min_ens-3]*2, color='k', alpha=0.0)
    plt.xlabel(r"Time [s]")
    plt.ylabel(r"Ensemble")
    lgd = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0,  edgecolor='k', framealpha=1.0,)
    plt.savefig('pattern.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
