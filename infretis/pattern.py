import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})


def pattern(inp, cap=250):
    a = np.loadtxt(inp)[:cap]
    min0 = min(a[:, 1])
    a[:,1] -= min0
    a[:,2] -= min0
    
    workers = list(set(a[:, 3]))
    min_ens, max_ens = min(a[:,0]), max(a[:,0])
    c_dic = {0: 'r', 1: 'b', 2: 'g'}
    c_dic2 = {0: '#ffcccc', 1: '#ccccff', 2: '#cce5cc'}
    dic = {int(worker): None  for worker in workers}
    label_workers = []

    # plot worker end-> start |
    for worker in workers:
        x = np.array([(i) for i in a if i[3] == int(worker)])
        time_ends = list(set([i[2] for i in x]))
        enss = {time: [] for time in time_ends}
        for time in time_ends:
            for i in x:
                if i[1] in time_ends:
                    enss[i[1]].append(i[0])
                if i[2] in time_ends:
                    enss[i[2]].append(i[0])
        for time in time_ends:
            if len(set(enss[time])) > 1:
                minmax = [min(enss[time]), max(enss[time])]
                plt.plot([time]*2, minmax, '--',
                         color=c_dic2[int(worker)], linewidth='2.')
        dic[int(worker)] = time_ends

    # scatter
    for worker in workers:
        for i in dic[int(worker)]:
            enss = list(range(int(min_ens), int(max_ens)+1))
            time = i
            for row in a:
                if row[1] < time < row[2] and worker != row[3]:
                    enss.pop(enss.index(int(row[0])))
            plt.scatter([time]*len(enss), enss,
                        color=c_dic2[int(worker)], marker=r's')

    # plot worker start -> end -
    for i in range(len(a)):
        worker = int(a[i][3])
        if worker not in label_workers:
            plt.plot([a[i][1], a[i][2]], [a[i][0]]*2, color=c_dic[worker],
                     marker="|", label=f'worker {worker}', linewidth='2.')
            label_workers.append(worker)
        else: 
            plt.plot([a[i][1], a[i][2]], [a[i][0]]*2, color=c_dic[worker],
                     marker="|", linewidth='2.')
    
    plt.xlabel(r"Time")
    plt.ylabel(r"Ensemble")
    plt.xticks([])
    plt.yticks(list(range(int(min_ens), int(max_ens)+1)))
    lgd = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
                     borderaxespad=0,  edgecolor='k', framealpha=1.0,)
    plt.savefig('pattern.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
