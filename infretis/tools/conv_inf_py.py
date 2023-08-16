import numpy as np

HEAD = "#     Step    No.-acc  No.-shoot l m r  Length Acc Mc            Min-O            Max-O Idx-Min Idx-Max          O-shoot  Idx-sh Idx-shN  Weight"


def read_infinity(inp):
    path_dic = {}
    with open(inp, "r") as read:
        for idx, line in enumerate(read):
            strip = line.rstrip()
            if "#" in line:
                continue
            split = strip.split()
            path_no = int(split[0])
            path_dic[path_no] = {}
            path_dic[path_no]["len"] = int(split[1])
            path_dic[path_no]["max_op"] = float(split[2])
            split_len = int(len(split[3:]) / 2)
            for idx, (frac, weight) in enumerate(
                zip(
                    split[3 : 3 + split_len],
                    split[3 + split_len : 3 + split_len * 2],
                )
            ):
                if "-" not in frac or "-" not in weight:
                    path_dic[path_no][f"{idx:03.0f}"] = (
                        np.float128(frac),
                        float(weight),
                    )
    return path_dic, split_len


def print_pathens(inp):
    dic, ens_len = read_infinity(inp)
    keys = list(dic.keys())
    rang = range(min(keys), max(keys))
    enss = [f"{i:03.0f}" for i in range(ens_len)]
    steps = {ens: 0 for ens in enss}
    moves = {ens: "sh" if ens in ["000", "001"] else "wf" for ens in enss}

    for ens in enss:
        with open(f"./{ens}/pathensemble.txt", "w") as fp:
            fp.write(HEAD + "\n")

    for i in keys:
        for ens in enss:
            if ens in dic[i].keys():
                pline = f"{steps[ens]:10.0f} {steps[ens]:10.0f} {steps[ens]:10.0f}"
                pline += " R R R \t"
                pline += f"  {dic[i]['len']:4.0f} ACC {moves[ens]}"
                pline += f" {0:2.10e}"
                pline += f" {dic[i]['max_op']:2.10e} {0:7.0f} {0:7.0f}"
                pline += f" {0:2.10e} {0:7.0f} {0:7.0f}"
                if dic[i][ens][0] == 0:
                    print(dic[i])
                    print("whadafa")
                    exit("ape")
                pline += f" {dic[i][ens][1]/dic[i][ens][0]:7.10e}"
                steps[ens] += 1
                if ens == "000":
                    with open(f"./{ens}/pathensemble.txt", "a") as fp:
                        for _ in range(int(dic[i][ens][0])):
                            fp.write(pline + "\n")
                else:
                    with open(f"./{ens}/pathensemble.txt", "a") as fp:
                        fp.write(pline + "\n")
