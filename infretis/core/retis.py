import logging
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

def retis_swap_zero(ensembles, settings, cycle):
    """Perform the RETIS swapping for ``[0^-] <-> [0^+]`` swaps.

    The RETIS swapping move for ensembles [0^-] and [0^+] requires some
    extra integration. Here we are generating new paths for [0^-] and
    [0^+] in the following way:

    1) For [0^-] we take the initial point in [0^+] and integrate
       backward in time. This is merged with the second point in [0^+]
       to give the final path. The initial point in [0^+] starts to the
       left of the interface and the second point is on the right
       side - i.e. the path will cross the interface at the end points.
       If we let the last point in [0^+] be called ``A_0`` and the
       second last point ``B``, and we let ``A_1, A_2, ...`` be the
       points on the backward trajectory generated from ``A_0`` then
       the final path will be made up of the points
       ``[..., A_2, A_1, A_0, B]``. Here, ``B`` will be on the right
       side of the interface and the first point of the path will also
       be on the right side.

    2) For [0^+] we take the last point of [0^-] and use that as an
       initial point to generate a new trajectory for [0^+] by
       integration forward in time. We also include the second last
       point of the [0^-] trajectory which is on the left side of the
       interface. We let the second last point be ``B`` (this is on the
       left side of the interface), the last point ``A_0`` and the
       points generated from ``A_0`` we denote by ``A_1, A_2, ...``.
       Then the resulting path will be ``[B, A_0, A_1, A_2, ...]``.
       Here, ``B`` will be on the left side of the interface and the
       last point of the path will also be on the left side of the
       interface.

    Parameters
    ----------
    ensembles : list of dictionaries of objects
        This is a list of the ensembles we are using in the RETIS method.
        It contains:

        * `path_ensemble`: object like :py:class:`.PathEnsemble`
          This is used for storing results for the simulation. It
          is also used for defining the interfaces for this simulation.
        * `system`: object like :py:class:`.System`
          System is used here since we need access to the temperature
          and to the particle list.
        * `order_function`: object like :py:class:`.OrderParameter`
          The class used for calculating the order parameter(s).
        * `engine`: object like :py:class:`.EngineBase`
          The engine to use for propagating a path.

    settings : dict
        This dict contains the settings for the RETIS method.
    cycle : integer
        The current cycle number.

    Returns
    -------
    out : string
        The result of the swapping move.

    """
    path_ensemble0 = ensembles[0]['path_ensemble']
    path_ensemble1 = ensembles[1]['path_ensemble']
    engine0, engine1 = ensembles[0]['engine'], ensembles[1]['engine']
    maxlen0 = settings['ensemble'][0]['tis']['maxlength']
    maxlen1 = settings['ensemble'][1]['tis']['maxlength']

    ens_moves = [settings['ensemble'][i]['tis'].get('shooting_move', 'sh')
                 for i in [0, 1]]
    intf_w = [list(i) for i in (path_ensemble0.interfaces,
                                path_ensemble1.interfaces)]
    for i, j in enumerate([settings['ensemble'][k] for k in (0, 1)]):
        if ens_moves[i] == 'wf':
            intf_w[i][2] = j['tis'].get('interface_cap', intf_w[i][2])

    # 0. check if MD is allowed
    allowed = (path_ensemble0.last_path.get_end_point(
                path_ensemble0.interfaces[0],
                path_ensemble0.interfaces[-1]) == 'R')
    if allowed:
        swap_ensemble_attributes(ensembles[0], ensembles[1], settings)
    # 1. Generate path for [0^-] from [0^+]:
    # We generate from the first point of the path in [0^+]:
    logger.debug('Swapping [0^-] <-> [0^+]')
    logger.debug('Creating path for [0^-]')
    system = path_ensemble1.last_path.phasepoints[0].copy()
    logger.debug('Initial point is: %s', system)
    ensembles[0]['system'] = system
    # Propagate it backward in time:
    path_tmp = path_ensemble1.last_path.empty_path(maxlen=maxlen1-1)
    if allowed:
        logger.debug('Propagating for [0^-]')
        engine0.propagate(path_tmp, ensembles[0], reverse=True)
    else:
        logger.debug('Not propagating for [0^-]')
        path_tmp.append(system)
    path0 = path_tmp.empty_path(maxlen=maxlen0)
    for phasepoint in reversed(path_tmp.phasepoints):
        path0.append(phasepoint)
    # Add second point from [0^+] at the end:
    logger.debug('Adding second point from [0^+]:')
    # Here we make a copy of the phase point, as we will update
    # the configuration and append it to the new path:
    phase_point = path_ensemble1.last_path.phasepoints[1].copy()
    logger.debug('Point is %s', phase_point)
    engine1.dump_phasepoint(phase_point, 'second')
    path0.append(phase_point)
    if path0.length == maxlen0:
        path0.status = 'BTX'
    elif path0.length < 3:
        path0.status = 'BTS'
    elif ('L' not in set(path_ensemble0.start_condition) and
          'L' in path0.check_interfaces(path_ensemble0.interfaces)[:2]):
        path0.status = '0-L'
    else:
        path0.status = 'ACC'

    # 2. Generate path for [0^+] from [0^-]:
    logger.debug('Creating path for [0^+] from [0^-]')
    # This path will be generated starting from the LAST point of [0^-] which
    # should be on the right side of the interface. We will also add the
    # SECOND LAST point from [0^-] which should be on the left side of the
    # interface, this is added after we have generated the path and we
    # save space for this point by letting maxlen = maxlen1-1 here:
    path_tmp = path0.empty_path(maxlen=maxlen1-1)
    # We start the generation from the LAST point:
    # Again, the copy below is not needed as the propagate
    # method will not alter the initial state.
    system = path_ensemble0.last_path.phasepoints[-1].copy()
    if allowed:
        logger.debug('Initial point is %s', system)
        ensembles[1]['system'] = system
        logger.debug('Propagating for [0^+]')
        engine1.propagate(path_tmp, ensembles[1], reverse=False)
        # Ok, now we need to just add the SECOND LAST point from [0^-] as
        # the first point for the path:
        path1 = path_tmp.empty_path(maxlen=maxlen1)
        phase_point = path_ensemble0.last_path.phasepoints[-2].copy()
        logger.debug('Add second last point: %s', phase_point)
        engine0.dump_phasepoint(phase_point, 'second_last')
        path1.append(phase_point)
        path1 += path_tmp  # Add rest of the path.
    else:
        path1 = path_tmp
        path1.append(system)
        logger.debug('Skipping propagating for [0^+] from L')

    if path_ensemble1.last_path.get_move() != 'ld':
        path0.set_move('s+')
    else:
        path0.set_move('ld')

    if path_ensemble0.last_path.get_move() != 'ld':
        path1.set_move('s-')
    else:
        path1.set_move('ld')
    if path1.length >= maxlen1:
        path1.status = 'FTX'
    elif path1.length < 3:
        path1.status = 'FTS'
    else:
        path1.status = 'ACC'
    logger.debug('Done with swap zero!')

    # Final checks:
    accept = path0.status == 'ACC' and path1.status == 'ACC'
    status = 'ACC' if accept else (path0.status if path0.status != 'ACC' else
                                   path1.status)
    # High Acceptance swap is required when Wire Fencing are used
    if accept and settings['tis'].get('high_accept', False):
        if 'wf' in ens_moves:
            accept, status = high_acc_swap([path1, path_ensemble1.last_path],
                                           ensembles[0]['rgen'],
                                           intf_w[0],
                                           intf_w[1],
                                           ens_moves)

    for i, path, path_ensemble, flag in ((0, path0, path_ensemble0, 's+'),
                                         (1, path1, path_ensemble1, 's-')):
        if not accept and path.status == 'ACC':
            path.status = status

        # These should be 1 unless length of paths equals 3.
        # This technicality is not yet fixed. (An issue is open as a reminder)

        ens_set = settings['ensemble'][i]
        move = ens_moves[i]
        path.weight = compute_weight(path, intf_w[i], move)\
            if (ens_set['tis'].get('high_accept', False) and
                move in ('wf', 'ss')) else 1

        # if accept:
        #     path_ensemble.move_path_to_generate(path)
        # else:
        #     logger.debug("Rejected swap path in [0^%s], %s", flag[:-1], status)
        # path_ensemble.add_path_data(path, path.status, cycle=cycle)
        # if cycle % ens_set.get('output', {}).get('restart-file', 1) == 0:
        #     write_ensemble_restart(ensembles[i], settings['ensemble'][i])

    return accept, (path0, path1), status

def swap_ensemble_attributes(ens1, ens2, settings):
    """Inplace swapping of attributes between ensembles."""
    for attr in settings.get('simulation', dict()).get('swap_attributes', []):
        logger.debug("Swapping attribute '%s' between ensembles: "
                     "%s <-> %s.",
                     attr,
                     ens1['path_ensemble'].ensemble_name,
                     ens2['path_ensemble'].ensemble_name)
        old_attr = ens1[attr]
        ens1[attr] = ens2[attr]
        ens2[attr] = old_attr
