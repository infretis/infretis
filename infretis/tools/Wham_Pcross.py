import os

import numpy as np

from infretis.tools.rec_error import rec_block_errors
from infretis.tools.toolsWHAM import PcrossWHAM2, get_WHAMfactors


def run_analysis(inp_dic):
    CalcFE = inp_dic["fener"]
    ifile = inp_dic["data"]
    lambda_interfaces = [float(i) for i in inp_dic["intfs"]]
    lamres = float(inp_dic["lamres"])
    minblocks = int(inp_dic["nblock"])
    nskip = int(inp_dic["nskip"])
    folder = inp_dic["folder"]

    # the Cxy values of [0+] are stored in the i0plus-th
    # column (first coulumn is counted as column nr 0)
    i0plus = 4
    # Note that the above might be different for permeation
    # simulations where more information is stored
    lambdaA = lambda_interfaces[0]
    lambdaB = lambda_interfaces[-1]
    # number of interfaces
    nintf = len(lambda_interfaces)
    # number of plus-ensembles [i+]
    nplus_ens = nintf - 1
    # The sum of fractional sampling occurrences for each [i+] ensemble
    eta = [0.0] * nplus_ens
    # Generate a list of values
    lambda_values = [
        i * lamres
        for i in range(round(lambdaA / lamres), round(lambdaB / lamres) + 1)
    ]
    # v_alpha: to become the total crossing probability based on WHAM
    v_alpha = [0.0] * len(lambda_values)
    # This entry is not changed anymore
    v_alpha[0] = 1
    # u_alpha: to become the total crossing probability
    # based on single point matching
    u_alpha = [0.0] * len(lambda_values)
    u_alpha[0] = 1  # This entry is not changed anymore
    p_loc = [[0.0] * len(lambda_values) for _ in range(nplus_ens)]
    # p_loc=[ [0.,0.,0.,..], [0.,0.,0.,..],
    # [0.,0.,0.,..], ...]
    # the set of arrays for local crossing probabilities:
    # p_loc[0] gives the profile P_A(lambda | \lambda_0)
    # as function of lambda with resolution lamres, p_loc[1]
    # is the same for P_A(lambda | \lambda_1)  etc

    # Open the file
    with open(ifile) as file:
        # Initialize the matrix
        matrix = []
        # Loop through each line in the file
        for line in file:
            # Ignore comment lines
            if line.startswith("#"):
                continue
            # Remove any trailing whitespace or newlines
            line = line.strip()
            # Split the line into a list of values
            values = line.split()
            # Replace any "----" values with 0.0
            values = [float(x) if x != "----" else 0.0 for x in values]
            # Add the row to the matrix
            matrix.append(values)
    # delete the first nskip entries
    del matrix[:nskip]

    ##check matrix
    # from checkm import *
    # check_matrix(matrix,nintf)

    # Format of data-file is not yet unweighted with HA-weights
    # column for [0-] ensemble
    i0min = i0plus - 1
    # sum of inverse HA-weights
    suminvw = [0.0] * nintf
    # sum of Pxy for all y being [0-], [0+], [1+] etc
    sumPxy = [0.0] * nintf
    # sum of Pxy after weighting with inverse HA-weights
    sumPxy_afterw = [0.0] * nintf
    for x in matrix:
        for y in range(nintf):
            # index of value that requires unweighting
            y1 = i0min + y
            # index of HA-weight
            y2 = y1 + nintf
            if x[y2] > 0:  # non-zero weight
                sumPxy[y] += x[y1]  # sum before unweighting
                x[y1] /= x[y2]
                sumPxy_afterw[y] += x[y1]  # sum after unweighting
            elif x[y1] > 0:
                print("Division by zero for path x=", x)
                exit()
            # sumPxy and suminvw remain the same.
            # if x[y1]==x[y2]==0, giving 0/0, x[y1] remains zero.
            # Next:
            # we used to store the HA-weights
            # store the running sum of sumPxy in the matrix where
            # running SumPxy values are needed for
            # computing the running WHAM averages
            # HA-weights are no longer needed after this step, while
            x[y2] = sumPxy[y]
    # loop over x completed
    # dvide each column by the average of the inverse HA-weight
    # This gives more comparible eta[y] values between ensembles.
    # For instance if [0+] is based on shooting and [i+] is based on WF
    # This allows to use the standard WHAM procedure
    for y in range(nintf):
        AvinvwHA = sumPxy_afterw[y] / sumPxy[y]
        y1 = i0min + y  # index of [0-], [0+], [1+] etc
        for x in matrix:
            x[y1] /= AvinvwHA

    # Same as WHAM_PQ, but this one allows us to compute the running average
    def WHAM_Ptot_run(interfaces, ploc_matrix, sumPxy):
        # ploc_matrix[j][i] equals P_A(\lambda_i|\lambda_j) as
        # obtained from the [j+] ensemble using the data so far
        n = len(interfaces) - 1  # \lambda_n=\lambda_B
        Qarray = [0.0] * n
        # the case [0+]
        P = 1.0
        invQ = sumPxy[0] * P
        if invQ == 0:
            return 0.0, Qarray
        Q = 1.0 / invQ
        Qarray[0] = Q
        for i in range(
            1, n + 1
        ):  # calculating P_A(\lambda_i|\lambda_0) upto i=n
            nominator = 0.0
            for j in range(i):
                nominator += sumPxy[j] * ploc_matrix[j][i]
            P = nominator * Q
            if i == n or P == 0:
                return P, Qarray
            # update invQ and Q for next round, if there is a next round
            invQ += sumPxy[i] / P
            Q = 1.0 / invQ
            Qarray[i] = Q

    # This function gives back the WHAM based crossing
    # probabilities at the interfaces and the Q-factor
    # to normailze the v_alpha vector
    def WHAM_PQ(npe, interf, res, eta, v_alpha):
        P = [0.0] * npe  # Crossin probability at lambda_0, lambda_1,
        #  .... lambda_{n-1}: P_A(\lambda_i|\lambda_0) based on WHAM
        Q = [0.0] * npe  # The Q-factor for Whamming (JCTC 2015 Lervik et al)
        invQ = [0.0] * npe  # The inverse Q-factor
        # Initial values
        P[0] = 1.0
        invQ[0] = eta[0]
        if invQ[0] == 0:
            return P, Q
        Q[0] = 1 / invQ[0]
        # Solve other values using recursive relations
        lambdaA = interf[0]
        for i in range(1, npe):  # Loop over all nterfaces after [0+]
            lambda_i = interf[i]
            alpha = round(
                (lambda_i - lambdaA) / res
            )  # the alpha index corresponding to lambda_i
            P[i] = v_alpha[alpha] * Q[i - 1]
            if P[i] == 0:
                return P, Q
            invQ[i] = invQ[i - 1] + (eta[i] / P[i])
            Q[i] = 1 / invQ[i]
        # add the final value of lambda_B
        P.append(v_alpha[-1] * Q[nplus_ens - 1])
        return P, Q

    # Loop over all paths x in the matrix to:
    # Compute eta[i] for all interfaces except the last lambda_B
    # Create the v_alpha() vector. Yet, without the proper normalization
    # Create the local crossing probabilities p_loc[.... ]
    # which is a matrix showing the local crossing probabilities
    # for [0+], [1+] etc
    # Create running average of local crossing probability
    # P_A(\lambda_{i+1}|\lambda_i): run_av_ploc
    # The structure of the data file is the following
    # x[0]=path-index, x[1]=path length, x[2]=lambda-max
    run_av_ploc = (
        []
    )  # running average for P_A(\lambda_{i+1}|\lambda_i) for i=0,1,2,...
    run_av_PtotWHAM = []
    run_av_Q = []
    ploc_runav_matrix = [
        [0.0] * nintf for _ in range(nintf)
    ]  # matrix where entry (i,j) gives P_A(\lambda_j|\lambda_i) for j>i
    for x in matrix:
        row_run_av_ploc = [0.0] * nplus_ens
        lambdamax = x[2]
        for i in range(nplus_ens):
            Cxy_index = i0plus + i
            Cxy = x[Cxy_index]
            eta[i] += Cxy  # increase eta[i]
            # Determine lower and upper bound for
            # increasing v_alpha- and p_loc values
            lambda_i = lambda_interfaces[i]
            alpha_max = int(np.floor((lambdamax - lambdaA) / lamres))
            alpha_min = round((lambda_i - lambdaA) / lamres)
            # Note: lambda_i-lambdaA)/lamres is an integer as
            # lambda_i and lambdaA should be commensurate with lamres
            if alpha_max > len(v_alpha) - 1:
                alpha_max = len(v_alpha) - 1  # -1 as we start counting from 0
            for alpha in range(alpha_min, alpha_max + 1):
                p_loc[i][alpha] += Cxy
            alpha_min += 1  # v(alpha) at the interface lambda_1, lambda_2
            # etc are determined by the previous [0+], [1+] etc
            for alpha in range(alpha_min, alpha_max + 1):
                v_alpha[alpha] += Cxy
            # update row i of ploc_runav_matrix
            ploc_runav_matrix[i][0:i] = [0.0] * i
            for j in range(i, nintf):
                lamj = lambda_interfaces[j]
                alpha_j = round(
                    (lamj - lambdaA) / lamres
                )  # Note again, interfaces are commensurate with lamres
                ploc_runav_matrix[i][j] = (
                    p_loc[i][alpha_j] / eta[i] if eta[i] != 0.0 else 0.0
                )
        row_run_av_ploc = [
            ploc_runav_matrix[row][row + 1]
            for row in range(len(ploc_runav_matrix) - 1)
        ]  # running average of P_A(\lambda_{i+1}|\lambda_i)
        run_av_ploc.append(row_run_av_ploc)
        sumPxy = x[-nplus_ens:]  # the sum of Pxy upto this path x
        Ptot_wham, Qarray = WHAM_Ptot_run(
            lambda_interfaces, ploc_runav_matrix, sumPxy
        )
        run_av_PtotWHAM.append(Ptot_wham)
        run_av_Q.append(Qarray)

    run_av_PtotPM = []
    # Get the running average for Ptot_pm:
    # total crossing probability based on point-matching
    for row in run_av_ploc:
        P_pm = np.prod(row)
        run_av_PtotPM.append(P_pm)
    # zip the two results together
    run_av_Ptot = zip(run_av_PtotPM, run_av_PtotWHAM)

    # Normalize the p_loc arrays such that each local
    # crossing probability of ensemble [i+] starts with 1 at lambda_i
    Pi0_wham, Q = WHAM_PQ(nplus_ens, lambda_interfaces, lamres, eta, v_alpha)
    print(
        "Check. The following two numbers should be the nearly same:",
        Pi0_wham[-1],
        Ptot_wham,
    )
    for i in range(nplus_ens):
        p_loc[i] = [val / eta[i] for val in p_loc[i]]

    # we need now a loop ovber all alpha values and determine K(alpha)
    # It is however faster to loop over K(alpha) indexes and split the
    # alpha-loop into parts that have the same K(alpha) value
    for Kalpha in range(nplus_ens):  # Loop over all pluse-ensembles after [0+]
        lambda_i = lambda_interfaces[Kalpha]
        lambda_next = lambda_interfaces[Kalpha + 1]
        alpha_max = round((lambda_next - lambdaA) / lamres)
        # Note again (lambda_next-lambdaA)/lamres is an integer
        # as interfaces should be commensurate with lamres resolution
        alpha_min = (
            round((lambda_i - lambdaA) / lamres) + 1
        )  # +1 because K(lambda) refers to index K with lambda_K < lambda
        # (not less or equal)
        for alpha in range(alpha_min, alpha_max + 1):
            v_alpha[alpha] *= Q[Kalpha]
    # v_alpha now represents the total crosing probability based on WHAM

    # Do same for single-point matching
    Pi0_pm = [
        0.0
    ] * nintf  # P_A(\lambda_i|\lambda_0) based on single point matching
    Pi0_pm[0] = 1.0
    for i in range(nplus_ens):
        lambda_i = lambda_interfaces[i]
        alpha_min = (
            round((lambda_i - lambdaA) / lamres) + 1
        )  # one grid-point next to lambda_i
        lambda_ip1 = lambda_interfaces[i + 1]
        alpha_max = round(
            (lambda_ip1 - lambdaA) / lamres
        )  # the grid-point exactly at lambda_(i+1)
        u_alpha[alpha_min : alpha_max + 1] = [
            Pi0_pm[i] * num for num in p_loc[i][alpha_min : alpha_max + 1]
        ]  # +1 because it is including alpha_max
        Pi0_pm[i + 1] = u_alpha[alpha_max]

    # Get running averages path lengths in [0-] and [0+]
    runav_L0min = []
    runav_L0plus = []
    sumL0min = 0.0
    sumL0plus = 0.0
    sumEta0min = 0.0
    sumEta0plus = 0.0
    for x in matrix:
        iL = 1  # index where path length of x is stored
        L = x[iL]
        sumEta0min += x[i0min]
        sumEta0plus += x[i0plus]
        sumL0min += x[i0min] * L
        sumL0plus += x[i0plus] * L
        r0min = sumL0min / sumEta0min if sumEta0min != 0.0 else 0.0
        r0plus = sumL0plus / sumEta0plus if sumEta0plus != 0.0 else 0.0
        runav_L0min.append(r0min)
        runav_L0plus.append(r0plus)

    print("Path length L0+ from [0+] ensemble data equals: ", r0plus)

    def get_L0plus_by_WHAM(matrix, lambda_interfaces, i0plus, Q):
        iL = 1  # index where path length of x is stored
        imax = 2  # index where lambda_max is stored
        intfQ = lambda_interfaces[
            :-1
        ]  # interfaces for determining Q-index, i.e. without lambda_B
        numC = len(intfQ)  # number of Cxy values that need to be summed
        # Note, for a given lambda_max, this Q-index corresponds to the
        # largest TIS-interface except lambda_B that is lower than lambda_max
        LWHAM = 0.0

        for x in matrix:
            L = x[iL]
            lmax = x[imax]
            indexQ = max(
                (i for i, val in enumerate(intfQ) if val < lmax), default=-1
            )
            if indexQ == -1:
                if lmax == intfQ[0] and L > 2:
                    indexQ == 0
                    # round-off issue that should not lead to an exit
                else:
                    print(
                        "Error: lambda_max is lower or equal"
                        + " to all TIS interfaces"
                    )
                    print("data line=", x)
                    print("lmax=", lmax)
                    print("interfaces except last: ", intfQ)
                    exit()
            Qmax = Q[indexQ]
            sumC = sum(x[i0plus : i0plus + numC])
            LWHAM += Qmax * L * sumC
        return LWHAM

    L0plusWHAM = get_L0plus_by_WHAM(matrix, lambda_interfaces, i0plus, Q)
    print("Path length L0+ from WHAM equals: ", L0plusWHAM)
    print("Remove redundant routine!!!!!!!!")

    # Alternative approach
    WHAMfactors = get_WHAMfactors(matrix, lambda_interfaces, i0plus, Q)
    # gives for each path the \chi(X) factor from which we
    # can compute any ensemble average < property(X)  >_[0^+] as
    # sum_X A(X) Chi(X)
    Lvalues = [x[1] for x in matrix]
    L0plusWHAM2 = sum([x * y for x, y in zip(Lvalues, WHAMfactors)])
    # L0plusWHAM2=get_L0plus_by_WHAM2(matrix,WHAMfactors)
    print("again:", L0plusWHAM2)

    # Compute crossing probability once more via WHAMfactors
    lammaxval = [x[2] for x in matrix]
    v2_alpha = PcrossWHAM2(lammaxval, lambda_values, WHAMfactors)

    # This calculates L0 via WHAM as a running average
    def get_runavL0plusWHAM(matrix, lambda_interfaces, i0plus, run_av_Q):
        iL = 1  # index where path length of x is stored
        imax = 2  # index where lambda_max is stored
        intfQ = lambda_interfaces[
            :-1
        ]  # interfaces for determining Q-index, i.e. without lambda_B
        numC = len(intfQ)  # number of Cxy values that need to be summed
        numintf = numC + 1
        # Note, for a given lambda_max, this Q-index corresponds to the
        # largest TIS-interface except lambda_B that is lower than lambda_max
        runavL0plusWHAM = []
        matrixLHC = [
            [0 for _ in range(numC)] for _ in range(numC)
        ]  # entry (i,j) is the sum over x of L(x)H_i(x) Cxj
        sumCxj = [0.0] * numC
        for t, x in enumerate(matrix):
            Q_array = run_av_Q[t]
            eta_array = x[
                i0plus + numintf : i0plus + numintf + numC
            ]  # "number of samples in each plus ensemble"
            L = x[iL]
            lmax = x[imax]
            indexH = max(
                (i for i, val in enumerate(intfQ) if val < lmax), default=-1
            )
            if indexH == -1:
                if lmax == intfQ[0] and L > 2:
                    indexH == 0
                    # round-off issue that should not lead to an exit
                else:
                    print(
                        "Error: lambda_max is lower or equal"
                        + " to all TIS interfaces"
                    )
                    print("data line=", x)
                    print("lmax=", lmax)
                    print("interfaces except last: ", intfQ)
                    exit()
            # update matrixLH and sumCxj
            for j in range(numC):
                Cxj = x[i0plus + j]
                sumCxj[j] += Cxj
                matrixLHC[indexH][j] += Cxj * L
            # Now the outer-loops over i and j
            L0WHAM = 0.0
            for i in range(numC):
                Qi = Q_array[i]
                for j in range(numC):
                    etaj = eta_array[j]
                    if sumCxj[j] > 0:
                        L0WHAM += Qi * etaj * matrixLHC[i][j] / sumCxj[j]
            runavL0plusWHAM.append(L0WHAM)
        return runavL0plusWHAM

    runav_L0plusWHAM = get_runavL0plusWHAM(
        matrix, lambda_interfaces, i0plus, run_av_Q
    )
    print(
        "Recomputed L0+-WHAM result using the running average routine:",
        runav_L0plusWHAM[-1],
    )

    # running average of flux. The flux is 1/(tau[0-]+tau[0+]) where
    # tau refer to the path length without counting end-points
    runav_FLUX_CONV = [
        1.0 / (y1 + y2 - 4) if (y1 + y2 - 4) > 0 else 0
        for y1, y2 in zip(runav_L0min, runav_L0plus)
    ]  # -4 because of neglecting end-points
    runav_FLUXWHAM = [
        1.0 / (y1 + y2 - 4) if (y1 + y2 - 4) > 0 else 0
        for y1, y2 in zip(runav_L0min, runav_L0plusWHAM)
    ]
    runav_FLUX = zip(runav_FLUX_CONV, runav_FLUXWHAM)
    # running average of RATE
    runav_RATE_PM = [y1 * y2 for y1, y2 in zip(runav_FLUX_CONV, run_av_PtotPM)]
    runav_RATE_WHAM = [
        y1 * y2 for y1, y2 in zip(runav_FLUX_CONV, run_av_PtotWHAM)
    ]
    runav_RATE_WHAMWHAM = [
        y1 * y2 for y1, y2 in zip(runav_FLUXWHAM, run_av_PtotWHAM)
    ]
    runav_RATE = zip(runav_RATE_PM, runav_RATE_WHAM, runav_RATE_WHAMWHAM)

    # average path-length per ensemble
    avpaths = [0.0] * nintf
    iL = 1  # index where path length is stored
    colL = [row[iL] for row in matrix]
    for i in range(nintf):
        icol = i + i0min
        colEns = [row[icol] for row in matrix]
        nom = sum([x * y for x, y in zip(colL, colEns)])
        denom = sum(colEns)
        avpaths[i] = nom / denom if denom != 0 else 0.0

    # Path length distributions
    # Here, we use a bin-width of 1. This might not in all
    # cases give directly visually nice plots.
    # However, from these data it is straightforward to get
    # plottable distributions by various aååroaches:
    # integration over larger bin-widths, use kernels, or compute
    # the cummulative plots that can be fitted and
    # differentiated using Savitsky-Golay filtering
    maxval = round(max(colL))
    distrmatrix = [[0.0] * nintf for _ in range(maxval + 1)]
    for x in matrix:
        L = round(x[iL])
        for i in range(nintf):
            icol = i + i0min
            distrmatrix[L][i] += x[icol]
    # normalize each column
    # for i in range(nintf):
    # Compute the column sums
    column_sums = [sum(column) for column in zip(*distrmatrix)]
    # Normalize each column by dividing by its sum
    distrmatrix = [
        [
            element / column_sums[column_index]
            if column_sums[column_index] != 0
            else 0
            for column_index, element in enumerate(row)
        ]
        for row in distrmatrix
    ]

    #################################################################
    #                                Error Ananlysis
    #################################################################
    errploc = []
    statineffploc = []
    blockerrs_ploc = []
    for i in range(nplus_ens):
        runav_i = [value[i] for value in run_av_ploc]
        err_i, statineff_i, blockerrs_i = rec_block_errors(runav_i, minblocks)
        errploc.append(err_i)
        statineffploc.append(statineff_i)
        blockerrs_ploc.append(blockerrs_i)
    blockerrs_ploc = [list(row) for row in zip(*blockerrs_ploc)]

    errPtotPM, statineffPtotPM, blockerrs_PtotPM = rec_block_errors(
        run_av_PtotPM, minblocks
    )
    errPtotWHAM, statineffPtotWHAM, blockerrs_PtotWHAM = rec_block_errors(
        run_av_PtotWHAM, minblocks
    )
    errPtot = [errPtotPM, errPtotWHAM]
    statineffPtot = [statineffPtotPM, statineffPtotWHAM]
    blockerrs_Ptot = zip(blockerrs_PtotPM, blockerrs_PtotWHAM)

    errL0min, statineffL0min, blockerrs_L0min = rec_block_errors(
        runav_L0min, minblocks
    )
    errL0plus, statineffL0plus, blockerrs_L0plus = rec_block_errors(
        runav_L0plus, minblocks
    )
    (
        errL0plusWHAM,
        statineffL0plusWHAM,
        blockerrs_L0plusWHAM,
    ) = rec_block_errors(runav_L0plusWHAM, minblocks)
    errL0 = [errL0min, errL0plus, errL0plusWHAM]
    statineffL0 = [statineffL0min, statineffL0plus, statineffL0plusWHAM]
    blockerrs_L0 = zip(blockerrs_L0min, blockerrs_L0plus, blockerrs_L0plusWHAM)

    errFLUX_CONV, statineffFLUX_CONV, blockerrs_FLUX_CONV = rec_block_errors(
        runav_FLUX_CONV, minblocks
    )
    errFLUX_WHAM, statineffFLUX_WHAM, blockerrs_FLUX_WHAM = rec_block_errors(
        runav_FLUXWHAM, minblocks
    )
    errFLUX = [errFLUX_CONV, errFLUX_WHAM]
    statineffFLUX = [statineffFLUX_CONV, statineffFLUX_WHAM]
    blockerrs_FLUX = zip(blockerrs_FLUX_CONV, blockerrs_FLUX_WHAM)

    errRATEpm, statineffRATEpm, blockerrs_RATEpm = rec_block_errors(
        runav_RATE_PM, minblocks
    )
    errRATEWHAM, statineffRATEWHAM, blockerrs_RATEWHAM = rec_block_errors(
        runav_RATE_WHAM, minblocks
    )
    (
        errRATEWHAMWHAM,
        statineffRATEWHAMWHAM,
        blockerrs_RATEWHAMWHAM,
    ) = rec_block_errors(runav_RATE_WHAMWHAM, minblocks)
    errRATE = [errRATEpm, errRATEWHAM, errRATEWHAMWHAM]
    statineffRATE = [statineffRATEpm, statineffRATEWHAM, statineffRATEWHAMWHAM]
    blockerrs_RATE = zip(
        blockerrs_RATEpm, blockerrs_RATEWHAM, blockerrs_RATEWHAMWHAM
    )

    ##################################################
    #                                     Write output
    ##################################################
    #### Define subroutine to easily get different ways to
    #### scale the local crossing probabilities
    if not os.path.isdir(folder):
        os.mkdir(folder)

    def write_ploc(ofile, rescale, folder=folder):
        ofile = os.path.join(folder, ofile)
        # Open the output file in write mode
        with open(ofile, "w") as file:
            # Write the column headers
            file.write("#Lam")
            for i in range(len(p_loc)):
                file.write(f"\t[{i}+]")
            file.write("\n")
            # Write the values
            for j in range(len(p_loc[0])):
                file.write(str(lambda_values[j]))
                for i in range(len(p_loc)):
                    yval = p_loc[i][j] * rescale[i]
                    # file.write(f"\t{p_loc[i][j]}")
                    file.write(f"\t{yval}")
                file.write("\n")
        print(f"Local crossing probabilities written to {ofile}")

    print(
        "local crossing probabilties will be written as 1)"
        + " unscaled, 2) scaled using point-matching, 3) scaled using WHAM"
    )
    # local crossing probabilities without scaling
    ofile = "ploc_unscaled.txt"
    rescale = [1.0] * len(p_loc)
    write_ploc(ofile, rescale)
    # local crossing probabilities with point-match scaling
    ofile = "ploc_pointmatch.txt"
    rescale = Pi0_pm[:-1]
    write_ploc(ofile, rescale)
    # local crossing probabilities with WHAM scaling
    ofile = "ploc_WHAM.txt"
    rescale = Pi0_wham
    write_ploc(ofile, rescale)

    ofile = "Pcross.txt"
    # Write total crossing probability based on WHAM and point-matching
    with open(os.path.join(folder, ofile), "w") as outfile:
        outfile.write("#lam    P-wham   P-point-matching P-wham2 \n")
        # Iterate over the lists and write each
        # triple of values as a line to the file
        for a, b, c, d in zip(lambda_values, v_alpha, u_alpha, v2_alpha):
            outfile.write(f"{a}\t{b}\t{c}\t{d}\n")
    print(
        "Wham and single-point matching based total"
        + " crossing probability written in file:",
        ofile,
    )

    # Write interface positions to a file with first colum some grid
    # values between 0 and 1, and next colums containing each a
    # single value equal to lambda_i
    # This can be convenient to plot vertical line between
    # y=0 and y=1 at x=lambda_i positions
    output_file = os.path.join(folder, "interfaces.txt")
    ystart = 1.0e-9  # small value.
    # The bottom of the vertical lines. Cannot be zero as
    # that is inconvenient in log-scale plots
    yend = 1.0 + ystart
    ystep = 0.01  # we use serveral y-values such that vertical
    # lines do not dissappear in plots when we zoom-in
    yvalues = np.arange(
        ystart, yend, ystep
    )  # we need several y-values as vertical lines otherwise
    # dissappear in e.g. gnuplot if you zoom in
    with open(output_file, "w") as file:
        for y in yvalues:
            # Write the first row ibased on yvalues followed
            # values from array lambda_interfaces
            file.write(f"\t{y}")
            for value in lambda_interfaces:
                file.write(f"\t{value}")
            file.write("\n")
    print(f"Interfaces written to {output_file}")

    # running averages of ploc, =estimates for P_A(\lambda_{i+1}|\lambda_i)
    filename = os.path.join(folder, "runav_ploc.txt")
    with open(filename, "w") as file:
        for row_index, row in enumerate(run_av_ploc):
            file.write(f"{row_index}   ")
            file.write("   ".join(f"{col:.8f}" for col in row))
            file.write("\n")
    print("Running averages of ploc written at: ", filename)

    # running averages of total crossing probability
    # via WHAM and single-point matching
    filename = os.path.join(folder, "runav_Pcross.txt")
    with open(filename, "w") as file:
        file.write("#counter  P_point-match P_WHAM\n")
        for row_index, (y1, y2) in enumerate(run_av_Ptot):
            file.write(f"{row_index}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write("\n")
    print(
        "Running averages of total crossing probability for "
        + "WHAM and point-matching written at: ",
        filename,
    )

    # running averages of path lengths
    filename = os.path.join(folder, "runav_L0.txt")
    with open(filename, "w") as file:
        file.write("#counter    L0-    L0+   L0+WHAM\n")
        for counter, (y1, y2, y3) in enumerate(
            zip(runav_L0min, runav_L0plus, runav_L0plusWHAM)
        ):
            file.write(f"{counter}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write(f"\t{y3}")
            file.write("\n")
    print(
        "Running averages of path lengths of [0-] and [0+]"
        + "(conventional and WHAM) written at: ",
        filename,
    )

    # running average of flux
    filename = os.path.join(folder, "runav_flux.txt")
    with open(filename, "w") as file:
        file.write("#counter    flux-conv flux-wham\n")
        for counter, (y1, y2) in enumerate(runav_FLUX):
            file.write(f"{counter}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write("\n")
    print("Running averages of flux written at: ", filename)

    # running average of rate
    filename = os.path.join(folder, "runav_rate.txt")
    with open(filename, "w") as file:
        file.write("#counter  rate-pm rate-wham rate-whamwham \n")
        for counter, (y1, y2, y3) in enumerate(runav_RATE):
            file.write(f"{counter}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write(f"\t{y3}")
            file.write("\n")
    print("Running averages of rate written at: ", filename)

    # average path length per ensembl
    filename = os.path.join(folder, "pathlengths.txt")
    with open(filename, "w") as file:
        file.write(
            "#ensemble-index  path-length (index -1 refers"
            + " to [0-], index i to [i+] for i=0,1,..  \n"
        )
        for counter, y in enumerate(avpaths):
            file.write(f"{counter-1}   ")
            file.write(f"\t{y}")
            file.write("\n")
    print("Averages path lengths per ensemble written at: ", filename)

    # path length distributions for each ensemble
    filename = os.path.join(folder, "pathdistr.txt")
    with open(filename, "w") as file:
        # Write the column headers
        file.write("#ens ")
        file.write(f"\t[{0}-]")
        for i in range(nplus_ens):
            file.write(f"\t[{i}+]")
        file.write("\n")
        # Write the values
        for j in range(len(distrmatrix)):
            file.write(str(j))
            for i in range(nplus_ens):
                yval = distrmatrix[j][i]
                file.write(f"\t{yval}")
            file.write("\n")
    print(f"Distributions of path lengths written to {filename}")

    # errors
    filename = os.path.join(folder, "errploc.txt")
    with open(filename, "w") as file:
        file.write("#averaged rel-error  " + str(errploc) + "\n")
        file.write("#statistical inefficiency: " + str(statineffploc) + "\n")
        file.write("#block-length rel-errors [0+], [1+] etc \n")
        for counter, row in enumerate(blockerrs_ploc):
            file.write(f"{counter+1}   ")
            file.write("   ".join(f"{col:.8f}" for col in row))
            file.write("\n")
    print(f"Error Analysis for ploc written to {filename}")

    filename = os.path.join(folder, "errPtot.txt")  # Name of the output file
    with open(filename, "w") as file:
        file.write("#averaged rel-error PM, WHAM: " + str(errPtot) + "\n")
        file.write(
            "#statistical inefficiency PM, WHAM: " + str(statineffPtot) + "\n"
        )
        file.write(
            "#Please note that here reported statistical"
            + " inefficiencies might not be easily interpretable\n"
        )
        file.write("#block-length rel-error \n")
        for counter, (y1, y2) in enumerate(blockerrs_Ptot):
            file.write(f"{counter+1}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write("\n")
    print(f"Error Analysis for Ptot written to {filename}")

    filename = os.path.join(folder, "errL0.txt")  # Name of the output file
    with open(filename, "w") as file:
        file.write(
            "#averaged rel-error L0min, L0plus, L0plusWHAM: "
            + str(errL0)
            + "\n"
        )
        file.write(
            "#statistical inefficiency L0min, L0plus, L0plusWHAM: "
            + str(statineffL0)
            + "\n"
        )
        file.write(
            "#Please note that the statistical inefficiency"
            + "for L0plusWHAM might not be easily interpretable\n"
        )
        file.write("#block-length rel-error \n")
        for counter, (y1, y2, y3) in enumerate(blockerrs_L0):
            file.write(f"{counter+1}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write(f"\t{y3}")
            file.write("\n")
    print(f"Error Analysis for L0 path length written to {filename}")

    filename = os.path.join(folder, "errFLUX.txt")
    with open(filename, "w") as file:
        file.write("#averaged rel-error CONV, WHAM: " + str(errFLUX) + "\n")
        file.write(
            "#statistical inefficiency CONV, WHAM: "
            + str(statineffFLUX)
            + "\n"
        )
        file.write(
            "#Please note that here reported statistical "
            + "inefficiencies might not be easily interpretable\n"
        )
        file.write("#block-length rel-error \n")
        for counter, (y1, y2) in enumerate(blockerrs_FLUX):
            file.write(f"{counter+1}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write("\n")
    print(f"Error Analysis for flux written to {filename}")

    filename = os.path.join(folder, "errRATE.txt")
    with open(filename, "w") as file:
        file.write(
            "#averaged rel-error PM, WHAM, WHAMWHAM: " + str(errRATE) + "\n"
        )
        file.write(
            "#statistical inefficiency PM, WHAM, WHAM: "
            + str(statineffRATE)
            + "\n"
        )
        file.write(
            "#Please note that here reported statistical "
            + "inefficiencies might not be easily interpretable\n"
        )
        file.write("#block-length rel-error \n")
        for counter, (y1, y2, y3) in enumerate(blockerrs_RATE):
            file.write(f"{counter+1}   ")
            file.write(f"\t{y1}")
            file.write(f"\t{y2}")
            file.write(f"\t{y3}")
            file.write("\n")
    print(f"Error Analysis for rate written to {filename}")

    # Calculate Landau Free energy?
    if "CalcFE" in locals() and CalcFE:
        WHAMfactorsMIN = [x[i0min] for x in matrix]
        sumWM = sum(WHAMfactorsMIN)
        WHAMfactorsMIN = [val / sumWM for val in WHAMfactorsMIN]
        # "Semi"-WHAM factors for the [0^-] ensemble
        WFtot = [a + b for a, b in zip(WHAMfactorsMIN, WHAMfactors)]
        trajlabels = [int(x[0]) for x in matrix]
        from infretis.tools.Free_energy import calculate_free_energy

        calculate_free_energy(trajlabels, WFtot)

    # Finished!
