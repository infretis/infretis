import numpy as np


def rec_blocks_np(r):
    r_np = np.array(r)
    n = len(r_np)
    result = np.zeros(n, dtype=r_np.dtype)
    result += np.arange(1, n + 1) * r_np
    result[1:] -= np.arange(1, n) * r_np[:-1]
    return result


def rec_block_errors(runav, minblocks):
    maxbll = int(len(runav) / minblocks)  # maximum block length
    bestav = runav[-1]  # most accurate average we have
    rel_errors = []
    for bloclengtgh in range(1, maxbll + 1):
        n = bloclengtgh
        runav_red = runav[n - 1 :: n]  # take every nth value of the array
        blocks = rec_blocks_np(runav_red)
        sum_qudiff = np.sum((blocks - bestav) ** 2)
        nb = len(blocks)
        Aerr2 = sum_qudiff / (nb * (nb - 1))
        Aerr = np.sqrt(Aerr2)  # esitimate of absolute error
        Rerr = Aerr / bestav
        rel_errors.append(Rerr)
    second_half = rel_errors[len(rel_errors) // 2 :]
    half_av_err = np.mean(second_half)
    Nstatineff = (half_av_err / rel_errors[0]) ** 2
    return half_av_err, Nstatineff, rel_errors
