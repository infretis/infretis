import numpy as np
import os
import time
from datetime import datetime
import tomli_w
import pickle
import logging
from infretis.classes.formatter import get_log_formatter
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"


# Define some infRETIS infrastructure
def calculate_ensemble_prob_and_weight(ens, num, offset=0):
    keys = np.array(list(ens.keys()))
    values = np.array(list(ens.values()))
    results = np.zeros(len(keys[0]), dtype="float128")
    idx = np.argsort(values)
    for key, value in zip(keys[idx], values[idx]):
        # 1/key[num] in the following line is the MC reweighting
        results += (np.array([1 if i else 0 for i in key]) *
                    value/abs(key[num+offset]))
    if num < 0:
        offset -= 1
    if results[offset] == 0:
        raise ValueError(f"{ens},{num},{offset},{results}")

    return results/results[offset], results[offset]


def analyze_data(data, offset=0):
    total_prob = np.array([1.0], dtype="float128")
    overall_prob = [1.0]
    ensembles = np.sort(list(data.data.keys()))
    for ens in ensembles:
        if ens < 0:
            continue
        prob, _ = calculate_ensemble_prob_and_weight(data.data[ens], ens,
                                                     offset=offset)
        if ens+offset+1 >= len(prob):
            total_prob = 0
            overall_prob.append(0)
            break
        next_prob = prob[ens+offset+1]
        overall_prob.append(next_prob*total_prob[0])
        total_prob *= next_prob  # cross next interface
    return overall_prob, total_prob


class Results(object):
    def __init__(self, n=3, offset=0):
        self.data = {}
        self.avg_lengths = {}
        self._run_prob = {}
        self._run_weight = {}
        self._cycles = {}
        self._run_total_prob = []
        self._n = n
        self._offset = offset

    def update_ens(self, ens, traj, weight, length=None):
        e = self.data.get(ens, {})
        w = e.get(traj, 0)
        e[traj] = w+weight
        self.data[ens] = e
        if length is not None and weight != 0:
            # update pathlengths for flux calculation
            avg, total_weight = self.avg_lengths.get(ens, (0, 0))
            total_weight += weight
            avg += weight*(length-avg)/total_weight
            self.avg_lengths[ens] = (avg, total_weight)

    def update_run_prob(self, ens, n=0):
        run_p = self._run_prob.get(ens, [])
        run_w = self._run_weight.get(ens, [])
        run_c = self._cycles.get(ens, [])
        data = self.data.get(ens, None)
        if data is not None:
            temp = calculate_ensemble_prob_and_weight(data, ens,
                                                      offset=self._offset)

            prob, tot_weight = temp
            run_p.append(prob[ens+self._offset+1])
            run_w.append(tot_weight)
            run_c.append(n)
        self._run_prob[ens] = run_p
        self._run_weight[ens] = run_w
        self._cycles[ens] = run_c

    def update_run_total_prob(self):
        _, tot = analyze_data(self, self._offset)
        self._run_total_prob.append(tot)


class REPEX_state(object):
    def __init__(self, n=3, result=None, minus=False, workers=None):
        if minus:
            self._offset = int(minus)
            n += int(minus)
        else:
            self._offset = 0

        # set numpy random seed when we initiate REPEX_state
        np.random.seed(0)

        self.n = n
        self.state = np.zeros(shape=(n, n))
        self._locks = np.ones(shape=(n))
        self._last_prob = None
        self._random_count = 0
        self._n = 0
        self._trajs = ["" for i in range(n)]

        self.config = {}
        self.traj_num_dic = {}
        self.workers = workers
        self.cworker = None
        self.tsteps = None
        self.cstep = None
        self.screen = None
        self.mc_moves = []
        self.ensembles = {}
        self.engines = {}
        self.toinitiate = self.workers
        self.output_tasks = None
        self.data_file = None
        self.pattern_file = False
        self.locked0 = []
        self.locked = []
        self.pyretis_settings = None
        self.zeroswap = 0.5
        self.start_time = time.time()
        self.pstore = None

        # Not using this at the moment
        # self.result = Results(n, offset=self._offset)

    def pick_lock(self):
        if not self.locked0:
            return self.pick()
        else:
            self.locked0.pop()
        logger.info('pick locked!')
        enss = []
        trajs = [] 
        enss0, trajs0 = self.locked.pop()
        for ens, traj in zip(enss0, trajs0):
            enss.append(ens-self._offset)
            traj_idx = self.live_paths().index(int(traj))
            self.swap(traj_idx, ens)
            self.lock(ens)
            trajs.append(self._trajs[ens])
        if self.printing():
            self.print_pick(tuple(enss), tuple(trajs0), self.cworker)
        picked = {}
        for ens_num, inp_traj in zip(enss, inp_trajs):
            picked[ens_num] = {'ens': self.ensembles[ens_num],
                               'traj': inp_traj,
                               'pn_old': inp_traj.path_number,
                               'engine': self.engines[self.ensembles[ens_num].engine]}

        # return tuple(enss), tuple(trajs)
        return picked

    def prep_md_items(self, md_items):
        # pick/lock ens & path 
        md_items.pop('picked', None)
        if self.toinitiate >= 0:
            # assign pin
            md_items.update({'pin': self.cworker})
            # pick lock
            md_items['picked'] = self.pick_lock()
        else:
            md_items['picked'] = self.pick()

        # md_items['ens_nums'] = list(md_items['picked'].keys())
        # md_items['pnum_old'] = []
        # for key in md_items['picked'].keys():
        #     md_items['pnum_old'].append(md_items['picked'][key]['traj'].path_number)
        # md_items['pn_old'] = list(md_items['picked'])

        # allocate worker pin:
        if self.config['dask'].get('wmdrun', False):
            base = self.config['dask']['wmdrun'][md_items['pin']]
            md_items['md_worker'] = base

        # write pattern:
        if self.pattern_file and self.toinitiate == -1:
            self.write_pattern(md_items)
        else:
            md_items['md_start'] = time.time()

        # empty / update md_items:
        for key in ['moves', 'pnum_old', 'trial_len', 'trial_op', 'generated']:
            md_items[key] = []
        # md_items.update({'ens_nums': ens_nums})

        return md_items

    def pick(self):
        prob = self.prob.astype("float64").flatten()
        p = np.random.choice(self.n**2, p=np.nan_to_num(prob/np.sum(prob)))
        traj, ens = np.divmod(p, self.n)
        self.swap(traj, ens)
        self.lock(ens)
        traj = self._trajs[ens]
        # If available do 0+- swap with 50% probability

        ens_nums = (ens-self._offset,)
        inp_trajs = (traj,)

        if ((
             (ens == self._offset and not self._locks[self._offset-1]) or
             (ens == self._offset-1 and not self._locks[self._offset])
        ) and np.random.random() < self.zeroswap):
        # ) and 1):
            if ens == self._offset:
                # ens = 0
                other = self._offset - 1
                other_traj = self.pick_traj_ens(other)
                ens_nums = (-1, 0)
                inp_trajs = (other_traj, traj)
            else:
                # ens = -1
                other = self._offset
                other_traj = self.pick_traj_ens(other)
                ens_nums = (-1, 0)
                inp_trajs = (traj, other_traj)

        # lock and print the picked traj and ens
        pat_nums = [str(i.path_number) for i in inp_trajs]
        self.locked.append((list(ens_nums), pat_nums))
        if self.printing():
            self.print_pick(ens_nums, pat_nums, self.cworker)

        picked = {}
        for ens_num, inp_traj in zip(ens_nums, inp_trajs):
            picked[ens_num] = {'ens': self.ensembles[ens_num+1],
                               'traj': inp_traj,
                               'pn_old': inp_traj.path_number,
                               'engine': self.engines[self.ensembles[ens_num+1].engine]}
        return picked

    def pick_traj_ens(self, ens):
        prob = self.prob.astype("float64")[:, ens].flatten()
        traj = np.random.choice(self.n, p=np.nan_to_num(prob/np.sum(prob)))
        self.swap(traj, ens)
        self.lock(ens)
        return self._trajs[ens]

    def write_ensembles(self):
        out = self.prob
        out = out.T
        out = np.nan_to_num(out)
        for i, ens in enumerate(out):
            if self._locks[i]:
                continue
            for j, weight in enumerate(ens):
                if weight != 0:
                    traj = self.state[j]
                    path = self._trajs[j]
                    if i in [self._offset, self._offset-1]:
                        # ens 0 and -1
                        length = getattr(path, 'length', None)
                    else:
                        length = None
        #             self.result.update_ens(i-self._offset, tuple(traj), weight,
        #                                    length=length)
        #     self.result.update_run_prob(i-self._offset, n=self._n)
        # self.result.update_run_total_prob()

    def add_traj(self, ens, traj, valid, count=True, n=0):

        if ens >= 0 and self._offset != 0:
            valid = tuple([0 for _ in range(self._offset)] + list(valid))
        elif ens < 0:
            valid = tuple(list(valid) +
                          [0 for _ in range(self.n - self._offset)])
        ens += self._offset
        # print('dog a', ens, valid, traj.path_number, traj.length)
        assert valid[ens] != 0
        # invalidate last prob
        self._last_prob = None
        self._trajs[ens] = traj
        self.state[ens, :] = valid
        self.unlock(ens)

        if count:
            self.write_ensembles()
            self._n += 1

    def sort_trajstate(self):
        needstomove = [self.state[idx][:-1][idx] == 0 for idx in range(self.n-1)]
        while True in needstomove and self.toinitiate == -1:
            ens_idx = list(needstomove).index(True)
            locks = self.locked_paths()
            zero_idx = list(self.state[ens_idx][1:-1]).index(0) + 1
            avail = [1 if i != 0 else 0 for i in self.state[:, zero_idx]]
            avail = [j if self._trajs[i].path_number not in locks else 0 for i, j in enumerate(avail[:-1])]
            trj_idx =  avail.index(1)
            self.swap(ens_idx, trj_idx)
            needstomove = [self.state[idx][:-1][idx] == 0 for idx in range(self.n-1)]
        self._last_prob = None
        self.prob

    def lock(self, ens):
        # invalidate last prob
        self._last_prob = None
        assert self._locks[ens] == 0
        self._locks[ens] = 1

    def unlock(self, ens):
        # invalidate last prob
        self._last_prob = None
        assert self._locks[ens] == 1
        self._locks[ens] = 0

    def swap(self, traj, ens):
        # mainly to keep the locks symmetric
        self.state[[ens, traj]] = self.state[[traj, ens]].copy()
        temp1 = self._trajs[ens]
        self._trajs[ens] = self._trajs[traj]
        self._trajs[traj] = temp1

    def live_paths(self):
        return [traj.path_number for traj in self._trajs[:-1]]

    def locked_paths(self):
        locks = [t0.path_number for t0, l0 in
                 zip(self._trajs[:-1], self._locks[:-1]) if l0]
        return locks

    def save_rng(self):
        rng_dic = {'rng-state': np.random.get_state()}
        save_loc = self.config['simulation'].get('save_loc', './')
        save_loc = os.path.join('./', save_loc, 'infretis.restart')
        with open(save_loc, 'wb') as outfile:
            pickle.dump(rng_dic, outfile)

    def set_rng(self):
        save_loc = self.config['simulation'].get('save_loc', './')
        save_loc = os.path.join('./', save_loc, 'infretis.restart')
        with open(save_loc, 'rb') as infile:
            info = pickle.load(infile)
        np.random.set_state(info['rng-state'])

    def loop(self):
        if self.printing():
            if self.cstep not in (0, self.config['current'].get('restarted_from', 0)):
                logger.info('date: ' + datetime.now().strftime(DATE_FORMAT))
                logger.info(f'------- infinity {self.cstep:5.0f} END ------- ' + '\n')

        if self.cstep >= self.tsteps:
            # should probably add a check for stopping when all workers are free
            # to close the while loop, but for now when cstep >= tsteps we return
            # false.
            self.save_rng()
            self.print_end()
            self.write_toml()
            logger.info('date: ' + datetime.now().strftime(DATE_FORMAT))
            return False

        self.cstep += 1
        self.config['current']['cstep'] = self.cstep
        
        if self.printing() and self.cstep <= self.tsteps:
            logger.info(f'------- infinity {self.cstep:5.0f} START ------- ')
            logger.info('date: ' + datetime.now().strftime(DATE_FORMAT))

        return self.cstep <= self.tsteps

    def initiate(self):
        if not self.cstep < self.tsteps:
            return False

        self.cworker = self.workers - self.toinitiate

        if self.toinitiate == self.workers:
            if self.screen > 0:
                self.print_start()
        if self.toinitiate < self.workers:
            if self.screen > 0:
                logger.info(f'------- submit worker {self.cworker-1} END ------- ' +
                             datetime.now().strftime(DATE_FORMAT) + '\n')
        if self.toinitiate > 0:
            if self.screen > 0:
                logger.info(f'------- submit worker {self.cworker} START ------- ' +
                             datetime.now().strftime(DATE_FORMAT))
        self.toinitiate -= 1
        return self.toinitiate >= 0


    @property
    def prob(self):
        if self._last_prob is None:
            prob = self.inf_retis(abs(self.state), self._locks)
            self._last_prob = prob.copy()
        return self._last_prob

    def inf_retis(self, input_mat, locks):
        # Drop locked rows and columns
        bool_locks = locks == 1
        # get non_locked minus interfaces
        offset = self._offset - sum(bool_locks[:self._offset])
        # make insert list
        i = 0
        insert_list = []
        for lock in bool_locks:
            if lock:
                insert_list.append(i)
            else:
                i += 1

        # Drop locked rows and columns
        non_locked = input_mat[~bool_locks, :][:, ~bool_locks]

        # Sort based on the index of the last non-zero values in the rows
        # argmax(a>0) gives back the first column index that is nonzero
        # so looping over the columns backwards and multiplying by -1 gives the
        # right ordering
        minus_idx = np.argsort(np.argmax(non_locked[:offset] > 0, axis=1))
        pos_idx = (np.argsort(-1 *
                              np.argmax(non_locked[offset:, ::-1] > 0, axis=1)
                              )
                   + offset)

        sort_idx = np.append(minus_idx, pos_idx)
        sorted_non_locked = non_locked[sort_idx]

        # check if all trajectories have equal weights
        sorted_non_locked_T = sorted_non_locked.T
        # Check the minus interfaces
        equal_minus = np.all(sorted_non_locked_T[
            np.where(sorted_non_locked_T[:, :offset] !=
                     sorted_non_locked_T[offset-1, :offset])
            ] == 0)
        # check the positive interfaces
        if len(sorted_non_locked_T) <= offset:
            equal_pos = True
        else:
            equal_pos = np.all(sorted_non_locked_T[:, offset:][
                np.where(sorted_non_locked_T[:, offset:] !=
                         sorted_non_locked_T[offset, offset:])
                ] == 0)

        equal = equal_minus and equal_pos

        out = np.zeros(shape=sorted_non_locked.shape, dtype="float128")
        if equal:
            # All trajectories have equal weights, run fast algorithm
            # run_fast
            # minus move should be run backwards
            out[:offset, ::-1] = self.quick_prob(sorted_non_locked[:offset,
                                                                   ::-1])
            if offset < len(out):
                # Catch only minus ens available
                out[offset:] = self.quick_prob(sorted_non_locked[offset:])
        else:
            #TODO DEBUG print
            # print("DEBUG this should not happen outside of wirefencing")
            blocks = self.find_blocks(sorted_non_locked, offset=offset)
            for start, stop, direction in blocks:
                if direction == -1:
                    cstart, cstop = stop-1, start-1
                    if cstop < 0:
                        cstop = None
                else:
                    cstart, cstop = start, stop
                subarr = sorted_non_locked[start:stop, cstart:cstop:direction]
                subarr_T = subarr.T
                if len(subarr) == 1:
                    out[start:stop, start:stop] = 1
                elif np.all(subarr_T[np.where(subarr_T != subarr_T[0])] == 0):
                    # Either the same weight as the last one or zero
                    temp = self.quick_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp
                elif len(subarr) <= 12:
                    # We can run this subsecond
                    temp = self.permanent_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp
                else:
                    self._random_count += 1
                    #TODO DEBUG PRINTS
                    print(f"random #{self._random_count}, "
                          f"dims = {len(subarr)}")
                    # do n random parrallel samples
                    temp = self.random_prob(subarr)
                    out[start:stop, cstart:cstop:direction] = temp

        out[sort_idx] = out.copy()  # COPY REQUIRED TO NOT BRAKE STATE!!!

        # Make sure we have a valid probability square
        assert np.allclose(np.sum(out, axis=1), 1)
        assert np.allclose(np.sum(out, axis=0), 1)

        # reinsert zeroes for the locked ensembles
        final_out_rows = np.insert(out, insert_list, 0, axis=0)

        # reinsert zeroes for the locked trajectories
        final_out = np.insert(final_out_rows, insert_list, 0, axis=1)

        return final_out

    def find_blocks(self, arr, offset):
        if len(arr) == 1:
            return ((0, 1, 1))
        # Assume no zeroes on the diagonal or lower triangle
        temp_arr = arr.copy()
        # for counting minus blocks
        temp_arr[:offset, :offset] = arr[:offset, :offset].T
        temp_arr[offset:, :offset] = 1  # add ones to the lower triangle
        non_zero = np.count_nonzero(temp_arr, axis=1)
        blocks = []
        start = 0
        for i, e in enumerate(non_zero):
            if e == i + 1:
                direction = -1 if start < offset else 1
                blocks.append((start, e, direction))
                start = e
        return blocks

    def quick_prob(self, arr):
        total_traj_prob = np.ones(shape=arr.shape[0], dtype='float128')
        out_mat = np.zeros(shape=arr.shape, dtype='float128')
        working_mat = np.where(arr != 0, 1, 0)  # convert non-zero numbers to 1

        for i, column in enumerate(working_mat.T[::-1]):
            ens = column*total_traj_prob
            s = ens.sum()
            if s != 0:
                ens /= s
            out_mat[:, -(i+1)] = ens
            total_traj_prob -= ens
            # force negative values to 0
            total_traj_prob[np.where(total_traj_prob < 0)] = 0
        return out_mat

    def force_quick_prob(self, arr):
        # TODO: DEBUG CODE
        # ONLY HERE TO DEBUG THE OTHER MEHTODS
        total_traj_prob = np.ones(shape=arr.shape[0], dtype='float128')
        out_mat = np.zeros(shape=arr.shape, dtype='float128')

        force_arr = arr.copy()
        # Force everything to be identical
        force_arr[np.where(force_arr != 0)] = 1
        for i, column in enumerate(force_arr.T[::-1]):
            ens = column*total_traj_prob
            s = ens.sum()
            if s != 0:
                ens /= s
            out_mat[:, -(i+1)] = ens
            total_traj_prob -= ens
            # force negative values to 0
            total_traj_prob[np.where(total_traj_prob < 0)] = 0
        return out_mat

    def permanent_prob(self, arr):
        out = np.zeros(shape=arr.shape, dtype="float128")
        n = len(arr)
        for i in range(n):
            rows = [r for r in range(n) if r != i]
            sub_arr = arr[rows, :]
            for j in range(n):
                if arr[i][j] == 0:
                    continue
                columns = [r for r in range(n) if r != j]
                M = sub_arr[:, columns]
                f = self.fast_glynn_perm(M)
                out[i][j] = f*arr[i][j]
        return out/max(np.sum(out, axis=1))

    def random_prob(self, arr, n=10_000):
        out = np.eye(len(arr), dtype="float128")
        current_state = np.eye(len(arr))
        choices = len(arr)//2
        even = choices*2 == len(arr)

        # The probability to go right
        prob_right = np.nan_to_num(np.roll(arr, -1, axis=1)/arr)

        # The probability to go left
        prob_left = np.nan_to_num(np.roll(arr, 1, axis=1)/arr)

        start = 0
        zero_one = np.array([0, 1])
        p_m = np.array([1, -1])
        temp = np.where(current_state == 1)

        for i in range(n):
            direction = np.random.choice(p_m)
            if not even:
                start = np.random.choice(zero_one)

            temp_left = prob_left[temp]
            temp_right = prob_right[temp]

            if not even:
                start = np.random.choice(zero_one)

            if direction == -1:
                probs = (temp_left[start:-1:2] *
                         np.roll(temp_right, 1, axis=0)[start:-1:2])
            else:
                probs = temp_right[start:-1:2]*temp_left[start+1::2]

            r_nums = np.random.random(choices)
            success = r_nums < probs

            for j in np.where(success)[0]:
                idx = j*2+start
                temp_state = current_state[:, [idx+direction, idx]]
                current_state[:, [idx, idx+direction]] = temp_state
                temp_state_2 = temp[0][[idx+direction, idx]]
                temp[0][[idx, idx+direction]] = temp_state_2

            out += current_state

        return out/(n+1)

    def fast_glynn_perm(self, M):
        def cmp(a, b):
            if a == b:
                return 0
            elif a > b:
                return 1
            else:
                return -1
        row_comb = np.sum(M, axis=0, dtype='float128')
        n = len(M)

        total = 0
        old_grey = 0
        sign = +1

        binary_power_dict = {2**i: i for i in range(n)}
        num_loops = 2**(n-1)

        for bin_index in range(1, num_loops + 1):
            total += sign * np.multiply.reduce(row_comb)

            new_grey = bin_index ^ (bin_index // 2)
            grey_diff = old_grey ^ new_grey
            grey_diff_index = binary_power_dict[grey_diff]
            direction = 2 * cmp(old_grey, new_grey)
            if direction:
                new_vector = M[grey_diff_index]
                row_comb += new_vector * direction

            sign = -sign
            old_grey = new_grey

        return total//num_loops

    def write_toml(self, ens_sel=(), input_traj=()):
        self.config['current']['active'] = self.live_paths()
        locked_ep = []
        for tup in self.locked:
            locked_ep.append(([int(tup0 + self._offset) for tup0 in tup[0]], tup[1]))
        self.config['current']['locked'] = locked_ep

        # save accumulative fracs
        self.config['current']['frac'] = {}
        for key in sorted(self.traj_num_dic.keys()):
            fracs = [str(i) for i in self.traj_num_dic[key]['frac']]
            self.config['current']['frac'][str(key)] = fracs

        with open("./restart.toml", "wb") as f:
            tomli_w.dump(self.config, f)

    def write_pattern(self, md_items):
        md_start = time.time()
        ensnums = '-'.join([str(i+1) for i in md_items['ens_nums']])
        with open(self.pattern_file, 'a') as fp:
            fp.write(f"{md_items['pin']}\t\t"
                   + f"{md_items['md_start'] - self.start_time:8.8f}\t"
                   + f"{md_items['wmd_start'] - self.start_time:8.8f}\t"
                   + f"{md_items['wmd_end'] - self.start_time:8.8f}\t"
                   + f"{md_items['md_end'] - self.start_time:8.8f}\t"
                   + f"{md_start - self.start_time:8.8f}\t"
                   + f"{ensnums}\n")
        md_items['md_start'] = md_start

    def printing(self):
        return self.screen > 0 and np.mod(self.cstep, self.screen) == 0

    def print_pick(self, ens_nums, pat_nums, pin):
        if len(ens_nums) > 1 or ens_nums[0] == -1:
            move = 'sh'
        else:
            move = self.mc_moves[ens_nums[0]+1]
        ens_p = ' '.join([f'00{ens_num+1}' for ens_num in ens_nums])
        pat_p = ' '.join(pat_nums)
        logger.info(f'shooting {move} in ensembles: {ens_p} with paths:' \
                    f' {pat_p} and worker: {pin}')

    def print_shooted(self, md_items, pn_news):
        if self.printing():
            moves = md_items['moves']
            ens_nums = ' '.join([f'00{i+1}' for i in md_items['ens_nums']])
            pnum_old = ' '.join([str(i) for i in md_items['pnum_old']])
            pnum_new = ' '.join([str(i) for i in pn_news])
            trial_lens = ' '.join([str(i) for i in md_items['trial_len']])
            trial_ops = ' '.join([f'[{i[0]:4.4f} {i[1]:4.4f}]' for i in md_items['trial_op']])
            status = md_items['status']
            simtime = md_items['md_end'] - md_items['md_start']
            logger.info(f"shooted {' '.join(moves)} in ensembles: {ens_nums}" \
                        f' with paths: {pnum_old} -> {pnum_new}')
            logger.info('with status:' \
                        f' {status} len: {trial_lens} op: {trial_ops} and' \
                        f' worker: {self.cworker} total time: {simtime:.2f}')
            self.print_state()

    def print_start(self):
        logger.info('stored ensemble paths:')
        ens_num = self.live_paths()
        logger.info(' '.join([f'00{i}: {j},' for i, j in enumerate(ens_num)]) + '\n')
        self.print_state()

    def print_state(self):
        last_prob = True
        if type(self._last_prob) == type(None):
            self.prob
            last_prob = False

        logger.info('===')
        to_print = '\t'.join(['e'+ f'{i:03.0f}' for i in range(self.n-1)])
        logger.info(' xx |\t' + to_print)
        logger.info(' -- |     -----------------------------------')

        locks = self.locked_paths()
        oil = False
        for idx, live in enumerate(self.live_paths()):
            if live not in locks:
                to_print = f'p{live:02.0f} |\t'
                if self.state[idx][:-1][idx] == 0 or self._last_prob[idx][:-1][idx] < 0.001:
                    oil = True
                for prob in self._last_prob[idx][:-1]:
                    to_print += f'{prob:.2f}\t' if prob != 0 else '----\t'
                to_print += f"{self.traj_num_dic[live]['max_op'][0]:8.5f} |"
                to_print += f"{self.traj_num_dic[live]['length']:5.0f}"
                logger.info(to_print)
            else:
                to_print = f'p{live:02.0f} |\t'
                logger.info(to_print + '\t'.join(['----' for j in range(self.n-1)]))
        if oil:
            logger.info('olive oil')
            oil = False

        logger.info('===')
        if not last_prob:
            self._last_prob = None

    def print_end(self):
        live_trajs = self.live_paths()
        stopping = self.cstep
        traj_num_dic = self.traj_num_dic
        logger.info('--------------------------------------------------')
        logger.info(f'live trajs: {live_trajs} after {stopping} cycles')
        logger.info('==================================================')
        logger.info('xxx | 000        001     002     003     004     |')
        logger.info('--------------------------------------------------')
        for key, item in traj_num_dic.items():
            values = '\t'.join([f'{item0:02.2f}' if item0 != 0.0 else '----' for item0 in item['frac'][:-1]])
            logger.info(f'{key:03.0f} * {values} *')

