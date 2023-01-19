import numpy as np
import matplotlib.pyplot as plt

def read(inp):
    max_op = []
    with open(inp, 'r') as read:
        for line in read:
            if '#' in line:
                continue
            split = line.rstrip().split()
            # max_op.append(float(split[10]))
            max_op.append(float(split[10]) > -0.7)
            # max_op.append(float(split[-1]))
            # print(split)
            # exit('hi')
    return max_op




corr = read('./retis_002.txt')
inco = read('./pathensemble.txt')
inc2 = read('./incorr.txt')

print(sum(corr[:2000]))
print(sum(inco[:2000]))
print(sum(inc2[:2000]))
# x1 = list(range(len(corr)))
# x2 = list(range(len(inco)))
# x3 = list(range(len(inc2)))
# print(np.average(corr))
# print(np.average(inco))
# plt.scatter(x1, corr, label='corr')
# plt.scatter(x2, inco, label='inco')
# plt.scatter(x3, inc2, label='inc2')
# plt.plot([x1[0], x1[-1]],[np.average(corr)]*2) 
# plt.plot([x2[0], x2[-1]],[np.average(inco)]*2) 
# plt.plot([x3[0], x3[-1]],[np.average(inc2)]*2) 
# plt.legend(frameon='false')
# plt.show()
# # read(corr)
