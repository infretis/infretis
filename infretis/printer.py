with open('infretis_data.txt', 'w') as fp:
    size = 6
    ens_str = '\t'.join([f'{i:03.0f}' for i in range(size)])
    fp.write('# ' + '-'*(34+8*size)+ '\n')
    fp.write('# ' + f'\txxx\tlen\tmax_OP\t\t{ens_str}\n')
    fp.write('# ' + '-'*(34+8*size) + '\n')

    pn_archive = [1,2,3,4,5]
    kaka1 = 209
    kaka2 = 0.27457
    zoop = [0.0, 2, 2, 2, 0.0]
    for pn in pn_archive:
        string = ''
        string += f'\t{pn:03.0f}\t'
        string += f"{kaka1:05.0f}" + '\t'
        string += f"{kaka2:05.5f}" + '\t\t'
        weight = '\t'.join([f'{item0:02.2f}' if item0 != 0.0 else '----' for item0 in zoop])
        fp.write(string + weight + '\t\n')

