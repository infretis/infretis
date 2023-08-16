import numpy as np


def npperm(M):
    n = M.shape[0]
    d = np.ones(n)
    j = 0
    s = 1
    f = np.arange(n)
    v = M.sum(axis=0)
    p = np.prod(v)
    while j < n - 1:
        v -= 2 * d[j] * M[j]
        d[j] = -d[j]
        s = -s
        prod = np.prod(v)
        p += s * prod
        f[0] = 0
        f[j] = f[j + 1]
        f[j + 1] = j + 1
        j = f[0]
    return p / 2 ** (n - 1)


# weight = np.array([[1., 1., 1., 0.,],
#                    [1., 1., 1., 0.,],
#                    [1., 1., 1., 1.,],
#                    [1., 1., 1., 1.,]])

weight = np.array(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]
)
frac = np.zeros(weight.shape)
print(weight)
print(" ")
for i in range(len(weight)):
    for j in range(len(weight)):
        row = list(range(len(weight)))
        col = list(range(len(weight)))
        row.pop(i)
        col.pop(j)
        weight[np.ix_(row, col)]
        frac[i, j] = (
            weight[i, j] * npperm(weight[np.ix_(row, col)]) / npperm(weight)
        )


print(" ")
print(weight)
print(frac)
# print(weight[np.ix_([0,1,2,3], [0,1,2,3])])
# print(npperm(weight))
