def get_WHAMfactors(matrix, lambda_interfaces, i0plus, Q):
    imax = 2  # index where lambda_max is stored
    intfQ = lambda_interfaces[
        :-1
    ]  # interfaces for determining Q-index, i.e. without lambda_B
    numC = len(intfQ)  # number of Cxy values that need to be summed
    # Note, for a given lambda_max, this Q-index corresponds to the
    # largest TIS-interface except lambda_B that is lower than lambda_max
    WHAMfactors = []
    for x in matrix:
        lmax = x[imax]
        indexQ = max(
            (i for i, val in enumerate(intfQ) if val < lmax), default=-1
        )
        if indexQ == -1:
            if lmax == intfQ[0]:
                indexQ == 0
                # round-off issue that should not lead to an exit
            else:
                print(
                    "Error: lambda_max is lower or equal to all TIS interfaces"
                )
                print("data line=", x)
                print("lmax=", lmax)
                print("interfaces except last: ", intfQ)
                exit()
        Qmax = Q[indexQ]
        sumC = sum(x[i0plus : i0plus + numC])
        Chi_X = Qmax * sumC
        WHAMfactors.append(Chi_X)
    return WHAMfactors


def PcrossWHAM2(lammaxval, lambda_values, WHAMfactors):
    v2_alpha = [0.0] * len(lambda_values)
    for lm, wf in zip(lammaxval, WHAMfactors):
        v2_alpha = [
            v2 + wf if i <= lm else v2
            for v2, i in zip(v2_alpha, lambda_values)
        ]
    return v2_alpha
