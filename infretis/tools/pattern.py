import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})


def pattern_reader(inp, cap=250):
    w_data = {}
    ensembles = []
    with open(inp) as read:
        for idx, line in enumerate(read):
            if "#" in line:
                continue
            split = line.rstrip().split()
            if len(w_data) == 0:
                float(split[1])
            float(split[1]) + float(split[5])
            key = int(split[0])
            if key not in w_data:
                w_data[key] = {
                    "md_start": [],
                    "wmd_start": [],
                    "wmd_end": [],
                    "md_end": [],
                    "dask_end": [],
                    "enss": [],
                    "connect": [],
                }

            float(split[1])
            w_data[key]["md_start"].append(float(split[1]))
            w_data[key]["wmd_start"].append(float(split[2]))
            w_data[key]["wmd_end"].append(float(split[3]))
            w_data[key]["md_end"].append(float(split[4]))
            w_data[key]["dask_end"].append(float(split[5]))
            w_data[key]["enss"].append([int(i) for i in split[6].split("-")])
            ensembles += [int(i) for i in split[6].split("-")]

            if len(w_data[key]["enss"]) <= 1:
                w_data[key]["connect"].append("-")
            else:
                eset = set(w_data[key]["enss"][-2] + w_data[key]["enss"][-1])
                connect = (
                    "-" if len(eset) == 1 else f"{min(eset)}-{max(eset)}"
                )
                w_data[key]["connect"].append(connect)
            if idx >= cap:
                break

    return w_data, list(set(ensembles))


def pattern(inp, cap=250, subtime=False, scatter=False):
    w_data, ensembles = pattern_reader(inp, cap)
    cols = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "k",
    ]
    c_dic = {i: j for i, j in enumerate(cols)}

    for worker in w_data.keys():
        start = w_data[worker]["md_start"][0]
        for idx, wstart in enumerate(w_data[worker]["md_start"]):
            wmds = w_data[worker]["wmd_start"][idx]
            wmde = w_data[worker]["wmd_end"][idx]
            mde = w_data[worker]["md_end"][idx]
            wend = w_data[worker]["dask_end"][idx]

            # horizontal lines
            for ens in w_data[worker]["enss"][idx]:
                plt.plot(
                    [wstart, wend],
                    [ens] * 2,
                    color=c_dic[worker],
                    linewidth="2.",
                )
                if subtime:
                    plt.scatter(
                        [wmds, wmde, mde],
                        [ens] * 3,
                        color="k",
                        marker="x",
                        linewidths=1.0,
                        alpha=1.0,
                    )
                plt.scatter(
                    [wstart, wend],
                    [ens] * 2,
                    color=c_dic[worker],
                    marker="|",
                    linewidth=2.0,
                )

            # vertical lines
            if w_data[worker]["connect"][idx] != "-":
                enss = w_data[worker]["connect"][idx].split("-")
                plt.plot(
                    [wstart] * 2,
                    [int(i) for i in enss],
                    "--",
                    color=c_dic[worker],
                    linewidth="2.",
                    alpha=0.4,
                )

        # scatter swap
        if scatter:
            other_w = [i for i in w_data.keys() if i != worker]
            for idx, d_end in enumerate(w_data[worker]["dask_end"]):
                free_ens = ensembles.copy()
                for ow in other_w:
                    for start, end, ens in zip(
                        w_data[ow]["md_start"],
                        w_data[ow]["dask_end"],
                        w_data[ow]["enss"],
                    ):
                        if start < d_end < end:
                            for ens0 in ens:
                                if ens0 in free_ens:
                                    free_ens.remove(ens0)
                plt.scatter(
                    [d_end] * len(free_ens),
                    free_ens,
                    color=c_dic[worker],
                    alpha=0.4,
                    marker=r"s",
                )

    plt.xlabel(r"Time")
    plt.ylabel(r"Ensemble")
    plt.xticks([])
    plt.yticks(list(range(min(ensembles), max(ensembles) + 1)))
    # plt.show()
    # lgd = plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",
    #                  borderaxespad=0,  edgecolor='k', framealpha=1.0,)
    # plt.savefig(
    #    "pattern.pdf",
    #    bbox_extra_artists=(lgd,),
    #    bbox_inches="tight"
    # )
    plt.savefig("pattern.pdf", bbox_inches="tight")
