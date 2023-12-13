import numpy as np


def rec_blocks(r):
    b = []
    for i in range(len(r)):
        if i == 0:
            b.append(r[i])
        else:
            b.append((i + 1) * r[i] - i * r[i - 1])
    return b


def rec_block_errors(runav, minblocks):
    maxbll = int(len(runav) / minblocks)  # maximum block length
    bestav = runav[-1]  # most accurate average we have
    rel_errors = []
    for bloclengtgh in range(1, maxbll + 1):
        n = bloclengtgh
        runav_red = runav[n - 1 :: n]  # take every nth value of the array
        blocks = rec_blocks(runav_red)
        sum_qudiff = sum(
            np.fromiter(((x - bestav) ** 2 for x in blocks), dtype=float)
        )
        nb = len(blocks)
        Aerr2 = sum_qudiff / (nb * (nb - 1))
        Aerr = np.sqrt(Aerr2)  # esitimate of absolute error
        Rerr = Aerr / bestav
        rel_errors.append(Rerr)
    second_half = rel_errors[len(rel_errors) // 2 :]
    half_av_err = np.mean(second_half)
    Nstatineff = (half_av_err / rel_errors[0]) ** 2
    return half_av_err, Nstatineff, rel_errors
