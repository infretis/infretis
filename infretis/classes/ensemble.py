"""Creates the ensemble dicts."""
from infretis.classes.rgen import create_random_generator


def create_ensembles(config):
    """Create all the ensemble dicts from the *toml config dict."""
    intfs = config['simulation']['interfaces']
    ens_intfs = []

    # set intfs for [0-] and [0+]
    ens_intfs.append([float('-inf'), intfs[0], intfs[0]])
    ens_intfs.append([intfs[0], intfs[0], intfs[-1]])

    # set interfaces and set detect for [1+], [2+], ...
    reactant, product = intfs[0], intfs[-1]
    for i in range(len(intfs)-2):
        middle = intfs[i + 1]
        ens_intfs.append([reactant, middle, product])

    # create all path ensembles
    pensembles = {}
    rgen = create_random_generator(settings={'seed':config['simulation']['seed']})
    for i, ens_intf in enumerate(ens_intfs):
        # #############RESTART SEED FROM RESTART...
        ens_seed = rgen.random_integers(1,9999999)
        print(i,ens_intf, ens_seed)
        rgen_ens = create_random_generator(settings={'seed':ens_seed})
        pensembles[i] = {'interfaces': tuple(ens_intf),
                         'tis_set': config['simulation']['tis_set'],
                         'mc_move': config['simulation']['shooting_moves'][i],
                         'eng_name': config['engine']['engine'],
                         'ens_name': f'{i:03d}',
                         'start_cond': 'R' if i == 0 else 'L',
                         'rgen': rgen_ens}
    return pensembles
