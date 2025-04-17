import numpy as np
import os
import tomli
HEAD = '#     Step    No.-acc  No.-shoot l m r  Length Acc Mc            Min-O            Max-O Idx-Min Idx-Max          O-shoot  Idx-sh Idx-shN  Weight'

def read_infinity(inp, rfile='restart.toml'):
    path_dic = {}
    with open(inp, 'r') as read:
        for idx, line in enumerate(read):
            strip = line.rstrip()
            if '#' in line:
                continue
            split = strip.split()
            path_no = int(split[0])
            path_dic[path_no] = {}
            path_dic[path_no]['len'] = int(split[1])
            path_dic[path_no]['max_op'] = float(split[2])
            path_dic[path_no]['min_op'] = float(split[3])
            path_dic[path_no]['ptype'] = split[4]
            split_len = int(len(split[5:])/2)
            print("split", split)
            print(split[5:5+split_len])
            print(split[5+split_len:5+split_len*2])
            for idx, (frac, weight) in enumerate(zip(split[5:5+split_len], split[5+split_len:5+split_len*2])):
                if '-' not in frac and '-' not in weight:
                    print("frac", frac)
                    print("weight", weight)
                    path_dic[path_no][f'{idx:03.0f}'] = (np.float128(frac), float(weight))

    userestart = False
    if os.path.isfile('restart.toml') and False:
        with open('./restart.toml', mode="rb") as f:
            restart = tomli.load(f)
        if restart.get('current', False):
            keys = restart['current'].keys()
            if all(i in keys for i in ['frac', 'weights', 'max_op', 'length']):
                userestart = True
                pns = list(restart['current']['frac'].keys())

    if userestart:
        dic_keys = list(path_dic.keys())
        for pn in pns:
            # make sure we are not overwriting existing pn
            ipn = int(pn)
            assert(ipn not in dic_keys)

            path_dic[ipn] = {}
            path_dic[ipn]['len'] = int(restart['current']['length'][pn])
            path_dic[ipn]['max_op'] = float(restart['current']['max_op'][pn])
            path_dic[ipn]['min_op'] = float(restart['current']['min_op'][pn])
            path_dic[ipn]['ptype'] = restart['current']['ptype'][pn]
            # to align weights with frac..
            weights = list(restart['current']['weights'][pn])
            if len(restart['current']['weights'][pn]) > 1:
                weights = [0.0] + weights
            for idx, (frac, weight) in enumerate(zip(restart['current']['frac'][pn],
                                                     weights)):
                if str(0.0) not in (frac, weight):
                    path_dic[ipn][f'{idx:03.0f}'] = (np.float128(frac), float(weight))

    return path_dic, split_len


def print_pathens(inp):
    dic, ens_len = read_infinity(inp)
    keys = list(dic.keys())
    rang = range(min(keys), max(keys))
    enss = [f'{i:03.0f}' for i in range(ens_len)]
    steps = {ens: 0 for ens in enss}
    moves = {ens: 'sh' if ens in ['000', '001'] else 'wf' for ens in enss}

    for ens in enss:
        with open(f'./{ens}/pathensemble.txt', 'w') as fp:
            fp.write(HEAD + '\n')

    for i in keys:
        for ens in enss:
            if ens in dic[i].keys():
                with open(f"./{ens}/mapperfile{ens}.txt","a") as mapperfile:
                    print(steps[ens],i,file=mapperfile)
                pline = f'{steps[ens]:10.0f} {steps[ens]:10.0f} {steps[ens]:10.0f} '
                pline += " ".join(dic[i]["ptype"])+"\t"
                pline += f"  {dic[i]['len']:4.0f} ACC {moves[ens]}"
                pline += f" {dic[i]['min_op']:2.10e}"
                pline += f" {dic[i]['max_op']:2.10e} {0:7.0f} {0:7.0f} {0:2.10e} {0:7.0f} {0:7.0f}"
                if dic[i][ens][0] == 0:
                    print(dic[i])
                    print('this should not happend, exit')
                    exit()
                # pline += f" {dic[i][ens][1]/dic[i][ens][0]:7.10e}"
                pline += f" 1.0000000000e+00"
                steps[ens] += 1
                if ens == '000' or True:
                    with open(f'./{ens}/pathensemble.txt', 'a') as fp:
                        for _ in range(int(dic[i][ens][0])):
                            fp.write(pline + '\n')
                # else:
                #         with open(f'./{ens}/pathensemble.txt', 'a') as fp:
                #             fp.write(pline + '\n')


print_pathens(inp="infretis_data.txt")
