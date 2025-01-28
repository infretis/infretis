"""Defines the main REPEX class for path handling and permanent calc."""

import logging
import os
import time
from datetime import datetime

import numpy as np
import tomli_w
from numpy.random import default_rng

from infretis.classes.engines.factory import assign_engines
from infretis.classes.formatter import PathStorage
from infretis.core.core import make_dirs
from infretis.core.tis import calc_cv_vector

logger = logging.getLogger("main")  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"

def spawn_rng(rgen: np.random.Generator) -> np.random.Generator:
    """
    Reimplementation of np.random.Generator.spawn() for numpy <= 1.24.4.

    Spawns a new random number generator (RNG) from an existing RNG.

    This function creates a new instance of the same type of RNG as the input
    RNG, using a seed generated from the input RNG's bit generator.

    Parameters:
    rgen (np.random.Generator): The input random number generator.

    Returns:
    np.random.Generator: A new random number generator instance.
    """
    return type(rgen)(type(rgen.bit_generator)(seed=rgen.bit_generator._seed_seq.spawn(1)[0]))

class REPEX_state:
    """Define the REPEX object."""

    # dicts to hold *toml, path data, ensembles and engine pointers.
    config: dict = {}
    traj_data: dict = {}
    ensembles: dict = {}
    engine_occ: dict = {}

    # holds counts current worker.
    cworker = None

    # sets simulation start time.
    start_time = time.time()

    # defines storage object.
    pstore = PathStorage()

    def __init__(self, config, minus=False):
        """Initiate REPEX given confic dict from *toml file."""
        self.config = config
        # storage of additional trajectory files
        self.pstore.keep_traj_fnames = config.get("output", {}).get(
            "keep_traj_fnames", []
        )
        # set rng
        if "restarted_from" in config["current"]:
            self.set_rgen()
        else:
            self.rgen = default_rng(seed=config.get("seed", 0))

        n = config["current"]["size"]
        if minus:
            self._offset = int(minus)
            n += int(minus)
        else:
            self._offset = 0

        self.n = n
        self.state = np.zeros(shape=(n, n))
        self._locks = np.ones(shape=n)
        self._last_prob = None
        self._random_count = 0
        self._trajs = [""] * n
        self.zeroswap = 0.5

        # detect any locked ens-path pairs exist pre start
        self.locked0 = list(self.config["current"].get("locked", []))
        self.locked = []

        # determines the number of initiation loops to do.
        self.toinitiate = self.workers

        # keep track of olds in case of delete_old = True
        self.pn_olds = {}

    @property
    def prob(self):
        """Calculate the P matrix."""
        if self._last_prob is None:
            prob = self.inf_retis(abs(self.state), self._locks)
            self._last_prob = prob.copy()
        return self._last_prob

    @property
    def cstep(self):
        """Retrieve cstep from config dict."""
        return self.config["current"]["cstep"]

    @cstep.setter
    def cstep(self, val):
        """Iterate += cstep from val."""
        self.config["current"]["cstep"] = val

    @property
    def tsteps(self):
        """Retrieve total steps from config dict."""
        return self.config["simulation"]["steps"]

    @property
    def screen(self):
        """Retrieve screen print frequency from config dict."""
        return self.config["output"]["screen"]

    @property
    def mc_moves(self):
        """Retrieve mc moves list from config dict."""
        return self.config["simulation"]["shooting_moves"]

    @property
    def cap(self):
        """Retrieve mc moves list from config dict."""
        return self.config["simulation"]["tis_set"].get("interface_cap", None)

    @property
    def pattern(self):
        """Retrieve pattern_file from config dict."""
        return self.config["output"].get("pattern", False)

    @property
    def data_dir(self):
        """Retrieve pattern_file from config dict."""
        return self.config["output"]["data_dir"]

    @property
    def pattern_file(self):
        """Retrieve pattern_file from config dict."""
        return self.config["output"]["pattern_file"]

    @property
    def data_file(self):
        """Retrieve data_file from config dict."""
        return self.config["output"]["data_file"]

    @property
    def interfaces(self):
        """Retrieve interfaces from config dict."""
        return self.config["simulation"]["interfaces"]

    @property
    def workers(self):
        """Retrieve workers from config dict."""
        return self.config["runner"]["workers"]

    def pick(self):
        """Pick path and ens."""
        prob = self.prob.astype("float64").flatten()
        p = self.rgen.choice(self.n**2, p=np.nan_to_num(prob / np.sum(prob)))
        traj, ens = np.divmod(p, self.n)
        self.swap(traj, ens)
        self.lock(ens)
        traj = self._trajs[ens]
        # If available do 0+- swap with 50% probability

        ens_nums = (ens - self._offset,)
        inp_trajs = (traj,)

        if (
            (ens == self._offset and not self._locks[self._offset - 1])
            or (ens == self._offset - 1 and not self._locks[self._offset])
        ) and self.rgen.random() < self.zeroswap:
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

        child_rng = spawn_rng(self.rgen)
        for ens_num, inp_traj in zip(ens_nums, inp_trajs):
            ens_pick = self.ensembles[ens_num + 1]
            ens_pick["rgen"] = spawn_rng(child_rng)
            picked[ens_num] = {
                "ens": ens_pick,
                "traj": inp_traj,
                "pn_old": inp_traj.path_number,
            }
        return picked

    def pick_traj_ens(self, ens):
        """Pick traj ens."""
        prob = self.prob.astype("float64")[:, ens].flatten()
        traj = self.rgen.choice(self.n, p=np.nan_to_num(prob / np.sum(prob)))
        self.swap(traj, ens)
        self.lock(ens)
        return self._trajs[ens]

    def pick_lock(self):
        """Pick path and ens.

        In case a crash, we pick lock locked from previous simulation.
        """
        if not self.locked0:
            if "restarted_from" in self.config["current"]:
                # get the same pick() as pre-restart. Need to set it again
                # because current self.rgen was used for calculating self.prob.
                self.set_rgen()
            return self.pick()

        enss = []
        trajs = []
        enss0, trajs0 = self.locked0.pop(0)
        logger.info("pick locked!")
        for ens, traj in zip(enss0, trajs0):
            enss.append(ens - self._offset)
            traj_idx = self.live_paths().index(int(traj))
            self.swap(traj_idx, ens)
            self.lock(ens)
            trajs.append(self._trajs[ens])
        if self.printing():
            self.print_pick(tuple(enss), tuple(trajs0), self.cworker)
        picked = {}

        child_rng = spawn_rng(self.rgen)
        for ens_num, inp_traj in zip(enss, trajs):
            ens_pick = self.ensembles[ens_num + 1]
            ens_pick["rgen"] = spawn_rng(child_rng)
            picked[ens_num] = {
                "ens": ens_pick,
                "traj": inp_traj,
                "pn_old": inp_traj.path_number,
            }
        return picked

    def prep_md_items(self, md_items):
        """Fill md_items with picked path and ens."""
        # Remove previous picked
        md_items.pop("picked", None)

        # pick/lock ens & path
        if self.toinitiate >= 0:
            # assign pin
            md_items.update({"pin": self.cworker})

            # pick lock
            md_items["picked"] = self.pick_lock()

            # set the worker folder
            w_folder = os.path.join(os.getcwd(), f"worker{md_items['pin']}")
            make_dirs(w_folder)
            md_items["w_folder"] = w_folder
        else:
            md_items["picked"] = self.pick()

        # Record ens_nums
        md_items["ens_nums"] = list(md_items["picked"].keys())

        # allocate worker pin:
        ens_engs = self.config["simulation"]["ensemble_engines"]
        eng_names = []
        for ens_num in md_items["ens_nums"]:
            md_items["picked"][ens_num]["exe_dir"] = md_items["w_folder"]
            if self.config["runner"].get("wmdrun", False):
                md_items["picked"][ens_num]["wmdrun"] = self.config["runner"][
                    "wmdrun"
                ][md_items["pin"]]
            # spawn rgen for all engines
            ens_rgen = md_items["picked"][ens_num]["ens"]["rgen"]
            md_items["picked"][ens_num]["rgen-eng"] = spawn_rng(ens_rgen)
            md_items["picked"][ens_num]["pin"] = md_items["pin"]
            eng_names += ens_engs[ens_num + 1]

        # engine assignment
        unique_eng_names = list(set(eng_names))
        eng_idx = assign_engines(
            self.engine_occ, unique_eng_names, md_items["pin"]
        )
        for ens_num in md_items["ens_nums"]:
            md_items["picked"][ens_num]["eng_idx"] = {
                eng: eng_idx[eng] for eng in ens_engs[ens_num + 1]
            }

        # write pattern:
        if self.pattern and self.toinitiate == -1:
            self.write_pattern(md_items)
        else:
            md_items["md_start"] = time.time()

        # record pnum_old
        md_items["pnum_old"] = []
        for key in md_items["picked"].keys():
            pnum_old = md_items["picked"][key]["traj"].path_number
            md_items["pnum_old"].append(pnum_old)

        # empty / update md_items:
        for key in ["moves", "trial_len", "trial_op", "generated"]:
            md_items[key] = []

        return md_items

    def add_traj(self, ens, traj, valid, count=True, n=0):
        """Add traj to state and calculate P matrix."""
        if ens >= 0 and self._offset != 0:
            valid = tuple([0 for _ in range(self._offset)] + list(valid))
        elif ens < 0:
            valid = tuple(
                list(valid) + [0 for _ in range(self.n - self._offset)]
            )
        ens += self._offset
        assert valid[ens] != 0
        # invalidate last prob
        self._last_prob = None
        self._trajs[ens] = traj
        self.state[ens, :] = valid
        self.unlock(ens)

        # Calculate P matrix
        self.prob

    def sort_trajstate(self):
        """Sort trajs and calculate P matrix."""
        needstomove = [
            self.state[idx][:-1][idx] == 0 for idx in range(self.n - 1)
        ]
        while True in needstomove and self.toinitiate == -1:
            ens_idx = list(needstomove).index(True)
            locks = self.locked_paths()
            zero_idx = list(self.state[ens_idx][1:-1]).index(0) + 1
            avail = [1 if i != 0 else 0 for i in self.state[:, zero_idx]]
            avail = [
                j if self._trajs[i].path_number not in locks else 0
                for i, j in enumerate(avail[:-1])
            ]
            trj_idx = avail.index(1)
            self.swap(ens_idx, trj_idx)
            needstomove = [
                self.state[idx][:-1][idx] == 0 for idx in range(self.n - 1)
            ]
        self._last_prob = None
        self.prob

    def lock(self, ens):
        """Lock ensemble."""
        # invalidate last prob
        self._last_prob = None
        assert self._locks[ens] == 0
        self._locks[ens] = 1

    def unlock(self, ens):
        """Unlock ensemble."""
        # invalidate last prob
        self._last_prob = None
        assert self._locks[ens] == 1
        self._locks[ens] = 0

    def swap(self, traj, ens):
        """Swap to keep the locks symmetric."""
        # mainly to keep the locks symmetric
        self.state[[ens, traj]] = self.state[[traj, ens]].copy()
        temp1 = self._trajs[ens]
        self._trajs[ens] = self._trajs[traj]
        self._trajs[traj] = temp1

    def live_paths(self):
        """Return list of live paths."""
        return [traj.path_number for traj in self._trajs[:-1]]

    def locked_paths(self):
        """Return list of locked paths."""
        locks = [
            t0.path_number
            for t0, l0 in zip(self._trajs[:-1], self._locks[:-1])
            if l0
        ]
        return locks

    def set_rgen(self):
        """Set numpy random generator state from restart."""
        seed_sequence = np.random.SeedSequence(
            entropy=0, n_children_spawned=self.cstep
        )
        self.rgen = default_rng(seed_sequence)
        self.rgen.bit_generator.state = self.config["current"]["rng_state"]

    def loop(self):
        """Check and iterate loop."""
        if self.printing():
            if self.cstep not in (
                0,
                self.config["current"].get("restarted_from", 0),
            ):
                logger.info("date: " + datetime.now().strftime(DATE_FORMAT))
                logger.info(
                    f"------- infinity {self.cstep:5.0f} END -------" + "\n"
                )

        if self.cstep >= self.tsteps:
            # should probably add a check for stopping when all workers
            # are free to close the while loop, but for now when
            # cstep >= tsteps we return false.
            self.print_end()
            self.write_toml()
            logger.info("date: " + datetime.now().strftime(DATE_FORMAT))
            return False

        self.cstep += 1

        if self.printing() and self.cstep <= self.tsteps:
            logger.info(f"------- infinity {self.cstep:5.0f} START -------")
            logger.info("date: " + datetime.now().strftime(DATE_FORMAT))

        return self.cstep <= self.tsteps

    def initiate(self):
        """Initiate loop."""
        if not self.cstep < self.tsteps:
            return False

        self.cworker = self.workers - self.toinitiate

        if self.toinitiate == self.workers:
            if self.screen > 0:
                self.print_start()
        if self.toinitiate < self.workers:
            if self.screen > 0:
                logger.info(
                    f"------- submit worker {self.cworker-1} END -------"
                    + datetime.now().strftime(DATE_FORMAT)
                    + "\n"
                )
        if self.toinitiate > 0:
            if self.screen > 0:
                logger.info(
                    f"------- submit worker {self.cworker} START -------"
                    + datetime.now().strftime(DATE_FORMAT)
                )
        self.toinitiate -= 1
        return self.toinitiate >= 0

    def inf_retis(self, input_mat, locks):
        """Permanent calculator."""
        # Drop locked rows and columns
        bool_locks = locks == 1
        # get non_locked minus interfaces
        offset = self._offset - sum(bool_locks[: self._offset])
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
        # so looping over the columns backwards and multiplying by -1
        # gives the right ordering
        minus_idx = np.argsort(np.argmax(non_locked[:offset] > 0, axis=1))
        pos_idx = (
            np.argsort(-1 * np.argmax(non_locked[offset:, ::-1] > 0, axis=1))
            + offset
        )

        sort_idx = np.append(minus_idx, pos_idx)
        sorted_non_locked = non_locked[sort_idx]

        # check if all trajectories have equal weights
        sorted_non_locked_T = sorted_non_locked.T
        # Check the minus interfaces
        equal_minus = np.all(
            sorted_non_locked_T[
                np.where(
                    sorted_non_locked_T[:, :offset]
                    != sorted_non_locked_T[offset - 1, :offset]
                )
            ]
            == 0
        )
        # check the positive interfaces
        if len(sorted_non_locked_T) <= offset:
            equal_pos = True
        else:
            equal_pos = np.all(
                sorted_non_locked_T[:, offset:][
                    np.where(
                        sorted_non_locked_T[:, offset:]
                        != sorted_non_locked_T[offset, offset:]
                    )
                ]
                == 0
            )

        equal = equal_minus and equal_pos

        out = np.zeros(shape=sorted_non_locked.shape, dtype="longdouble")
        if equal:
            # All trajectories have equal weights, run fast algorithm
            # run_fast
            # minus move should be run backwards
            out[:offset, ::-1] = self.quick_prob(
                sorted_non_locked[:offset, ::-1]
            )
            if offset < len(out):
                # Catch only minus ens available
                out[offset:] = self.quick_prob(sorted_non_locked[offset:])
        else:
            # TODO DEBUG print
            # print("DEBUG this should not happen outside of wirefencing")
            blocks = self.find_blocks(sorted_non_locked, offset=offset)
            for start, stop, direction in blocks:
                if direction == -1:
                    cstart, cstop = stop - 1, start - 1
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
                    # TODO DEBUG PRINTS
                    print(
                        f"random #{self._random_count}, "
                        f"dims = {len(subarr)}"
                    )
                    # do n random parallel samples
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
        """Find blocks in a W matrix."""
        if len(arr) == 1:
            return (0, 1, 1)
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
        """Quick P matrix calculation for specific W matrix."""
        total_traj_prob = np.ones(shape=arr.shape[0], dtype="longdouble")
        out_mat = np.zeros(shape=arr.shape, dtype="longdouble")
        working_mat = np.where(arr != 0, 1, 0)  # convert non-zero numbers to 1

        for i, column in enumerate(working_mat.T[::-1]):
            ens = column * total_traj_prob
            s = ens.sum()
            if s != 0:
                ens /= s
            out_mat[:, -(i + 1)] = ens
            total_traj_prob -= ens
            # force negative values to 0
            total_traj_prob[np.where(total_traj_prob < 0)] = 0
        return out_mat

    def permanent_prob(self, arr):
        """P matrix calculation for specific W matrix."""
        out = np.zeros(shape=arr.shape, dtype="longdouble")
        # Don't overwrite input arr
        scaled_arr = arr.copy()
        n = len(scaled_arr)
        # Rescaling the W-matrix avoids numerical instabilites when the
        # matrix is large and contains large weights from
        # high-acceptance moves
        for i in range(n):
            scaled_arr[i, :] /= np.max(scaled_arr[i, :])
        for i in range(n):
            rows = [r for r in range(n) if r != i]
            sub_arr = scaled_arr[rows, :]
            for j in range(n):
                if scaled_arr[i][j] == 0:
                    continue
                columns = [r for r in range(n) if r != j]
                M = sub_arr[:, columns]
                f = self.fast_glynn_perm(M)
                out[i][j] = f * scaled_arr[i][j]
        return out / max(np.sum(out, axis=1))

    def random_prob(self, arr, n=10_000):
        """P matrix calculation for specific W matrix."""
        out = np.eye(len(arr), dtype="longdouble")
        current_state = np.eye(len(arr))
        choices = len(arr) // 2
        even = choices * 2 == len(arr)

        # The probability to go right
        prob_right = np.nan_to_num(np.roll(arr, -1, axis=1) / arr)

        # The probability to go left
        prob_left = np.nan_to_num(np.roll(arr, 1, axis=1) / arr)

        start = 0
        zero_one = np.array([0, 1])
        p_m = np.array([1, -1])
        temp = np.where(current_state == 1)

        for i in range(n):
            direction = self.rgen.choice(p_m)
            if not even:
                start = self.rgen.choice(zero_one)

            temp_left = prob_left[temp]
            temp_right = prob_right[temp]

            if not even:
                start = self.rgen.choice(zero_one)

            if direction == -1:
                probs = (
                    temp_left[start:-1:2]
                    * np.roll(temp_right, 1, axis=0)[start:-1:2]
                )
            else:
                probs = temp_right[start:-1:2] * temp_left[start + 1 :: 2]

            r_nums = self.rgen.random(choices)
            success = r_nums < probs

            for j in np.where(success)[0]:
                idx = j * 2 + start
                temp_state = current_state[:, [idx + direction, idx]]
                current_state[:, [idx, idx + direction]] = temp_state
                temp_state_2 = temp[0][[idx + direction, idx]]
                temp[0][[idx, idx + direction]] = temp_state_2

            out += current_state

        return out / (n + 1)

    def fast_glynn_perm(self, M):
        """Glynn permanent."""

        def cmp(a, b):
            if a == b:
                return 0
            elif a > b:
                return 1
            else:
                return -1

        row_comb = np.sum(M, axis=0, dtype="longdouble")
        n = len(M)

        total = 0
        old_grey = 0
        sign = +1

        binary_power_dict = {2**i: i for i in range(n)}
        num_loops = 2 ** (n - 1)

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

        return total / num_loops

    def write_toml(self):
        """Toml writer."""
        self.config["current"]["active"] = self.live_paths()
        locked_ep = []
        for tup in self.locked:
            locked_ep.append(
                ([int(tup0 + self._offset) for tup0 in tup[0]], tup[1])
            )
        self.config["current"]["locked"] = locked_ep
        self.config["current"]["rng_state"] = self.rgen.bit_generator.state

        # save accumulative fracs
        self.config["current"]["frac"] = {}
        for key in sorted(self.traj_data.keys()):
            fracs = [str(i) for i in self.traj_data[key]["frac"]]
            self.config["current"]["frac"][str(key)] = fracs

        with open("./restart.toml", "wb") as f:
            tomli_w.dump(self.config, f)

    def write_pattern(self, md_items):
        """Pattern writer."""
        md_start = time.time()
        ensnums = "-".join([str(i + 1) for i in md_items["ens_nums"]])
        with open(self.pattern_file, "a") as fp:
            fp.write(
                f"{md_items['pin']}\t\t"
                + f"{md_items['md_start'] - self.start_time:8.8f}\t"
                + f"{md_items['wmd_start'] - self.start_time:8.8f}\t"
                + f"{md_items['wmd_end'] - self.start_time:8.8f}\t"
                + f"{md_items['md_end'] - self.start_time:8.8f}\t"
                + f"{md_start - self.start_time:8.8f}\t"
                + f"{ensnums}\n"
            )
        md_items["md_start"] = md_start

    def printing(self):
        """Check if print."""
        return self.screen > 0 and np.mod(self.cstep, self.screen) == 0

    def print_pick(self, ens_nums, pat_nums, pin):
        """Print pick."""
        if len(ens_nums) > 1 or ens_nums[0] == -1:
            move = "sh"
        else:
            move = self.mc_moves[ens_nums[0] + 1]
        ens_p = " ".join([f"00{ens_num+1}" for ens_num in ens_nums])
        pat_p = " ".join(pat_nums)
        logger.info(
            f"shooting {move} in ensembles: {ens_p} with paths:"
            f" {pat_p} and worker: {pin}"
        )

    def print_shooted(self, md_items, pn_news):
        """Print shooted."""
        moves = md_items["moves"]
        ens_nums = " ".join([f"00{i+1}" for i in md_items["ens_nums"]])
        pnum_old = " ".join([str(i) for i in md_items["pnum_old"]])
        pnum_new = " ".join([str(i) for i in pn_news])
        trial_lens = " ".join([str(i) for i in md_items["trial_len"]])
        trial_ops = " ".join(
            [f"[{i[0]:4.4f} {i[1]:4.4f}]" for i in md_items["trial_op"]]
        )
        status = md_items["status"]
        simtime = md_items["md_end"] - md_items["md_start"]
        logger.info(
            f"shooted {' '.join(moves)} in ensembles: {ens_nums}"
            f" with paths: {pnum_old} -> {pnum_new}"
        )
        logger.info(
            "with status:"
            f" {status} len: {trial_lens} op: {trial_ops} and"
            f" worker: {self.cworker} total time: {simtime:.2f}"
        )
        self.print_state()

    def print_start(self):
        """Print start."""
        logger.info("stored ensemble paths:")
        ens_num = self.live_paths()
        logger.info(
            " ".join([f"00{i}: {j}," for i, j in enumerate(ens_num)]) + "\n"
        )
        self.print_state()

    def print_state(self):
        """Print state."""
        last_prob = True
        if isinstance(self._last_prob, type(None)):
            self.prob
            last_prob = False

        logger.info("===")
        logger.info(" xx |\tv Ensemble numbers v")
        to_print = [f"{i:03.0f}" for i in range(self.n - 1)]
        for i in range(len(to_print[0])):
            to_print0 = " ".join([j[i] for j in to_print])
            if i == len(to_print[0]) - 1:
                to_print0 += "\t\tmax_op\tmin_op\tlen"
            logger.info(" xx |\t" + to_print0)

        logger.info(" -- |\t" + "".join("--" for _ in range(self.n + 14)))

        locks = self.locked_paths()
        oil = False
        for idx, live in enumerate(self.live_paths()):
            if live not in locks:
                to_print = f"p{live:02.0f} |\t"
                if (
                    self.state[idx][:-1][idx] == 0
                    or self._last_prob[idx][:-1][idx] < 0.001
                ):
                    oil = True
                for prob in self._last_prob[idx][:-1]:
                    if prob == 1:
                        marker = "x "
                    elif prob == 0:
                        marker = "- "
                    else:
                        marker = f"{int(round(prob*10,1))} "
                        # change if marker == 10
                        if len(marker) == 3:
                            marker = "9 "
                    to_print += marker
                to_print += f"|\t{self.traj_data[live]['max_op'][0]:5.3f} \t"
                to_print += f"{self.traj_data[live]['min_op'][0]:5.3f} \t"
                to_print += f"{self.traj_data[live]['length']:5.0f}"
                logger.info(to_print)
            else:
                to_print = f"p{live:02.0f} |\t"
                logger.info(
                    to_print + "".join(["- " for j in range(self.n - 1)]) + "|"
                )
        if oil:
            logger.info("olive oil")
            oil = False

        logger.info("===")
        if not last_prob:
            self._last_prob = None

    def print_end(self):
        """Print end."""
        live_trajs = self.live_paths()
        stopping = self.cstep
        logger.info("--------------------------------------------------")
        logger.info(f"live trajs: {live_trajs} after {stopping} cycles")
        logger.info("==================================================")
        logger.info("xxx | 000        001     002     003     004     |")
        logger.info("--------------------------------------------------")
        for key, item in self.traj_data.items():
            values = "\t".join(
                [
                    f"{item0:02.2f}" if item0 != 0.0 else "----"
                    for item0 in item["frac"][:-1]
                ]
            )
            logger.info(f"{key:03.0f} * {values} *")

    def treat_output(self, md_items):
        """Treat output."""
        pn_news = []
        md_items["md_end"] = time.time()
        picked = md_items["picked"]
        traj_num = self.config["current"]["traj_num"]

        for ens_num in picked.keys():
            pn_old = picked[ens_num]["pn_old"]
            out_traj = picked[ens_num]["traj"]
            self.ensembles[ens_num + 1] = picked[ens_num]["ens"]

            for idx, lock in enumerate(self.locked):
                if str(pn_old) in lock[1]:
                    self.locked.pop(idx)
            # if path is new: number and save the path:
            if out_traj.path_number is None or md_items["status"] == "ACC":
                # move to accept:
                ens_save_idx = self.traj_data[pn_old]["ens_save_idx"]
                out_traj.path_number = traj_num
                data = {
                    "path": out_traj,
                    "dir": os.path.join(
                        os.getcwd(), self.config["simulation"]["load_dir"]
                    ),
                }
                out_traj = self.pstore.output(self.cstep, data)
                self.traj_data[traj_num] = {
                    "frac": np.zeros(self.n, dtype="longdouble"),
                    "max_op": out_traj.ordermax,
                    "min_op": out_traj.ordermin,
                    "length": out_traj.length,
                    "weights": out_traj.weights,
                    "adress": out_traj.adress,
                    "ens_save_idx": ens_save_idx,
                }
                traj_num += 1
                if (
                    self.config["output"].get("delete_old", False)
                    and pn_old > self.n - 2
                ):
                    if len(self.pn_olds) > self.n - 2:
                        pn_old_del, del_dic = next(iter(self.pn_olds.items()))
                        # delete trajectory files
                        for adress in del_dic["adress"]:
                            os.remove(adress)
                        # delete txt files
                        load_dir = self.config["simulation"]["load_dir"]
                        if self.config["output"].get("delete_old_all", False):
                            for txt in ("order.txt", "traj.txt", "energy.txt"):
                                txt_adress = os.path.join(
                                    load_dir, pn_old_del, txt
                                )
                                if os.path.isfile(txt_adress):
                                    os.remove(txt_adress)
                            os.rmdir(
                                os.path.join(load_dir, pn_old_del, "accepted")
                            )
                            os.rmdir(os.path.join(load_dir, pn_old_del))
                        # pop the deleted path.
                        self.pn_olds.pop(pn_old_del)
                    # keep delete list:
                    if len(self.pn_olds) <= self.n - 2:
                        self.pn_olds[str(pn_old)] = {
                            "adress": self.traj_data[pn_old]["adress"],
                        }
            pn_news.append(out_traj.path_number)
            self.add_traj(ens_num, out_traj, valid=out_traj.weights)

        # record weights
        locked_trajs = self.locked_paths()
        if self._last_prob is None:
            self.prob
        for idx, live in enumerate(self.live_paths()):
            if live not in locked_trajs:
                self.traj_data[live]["frac"] += self._last_prob[:-1][idx, :]

        # write succ data to infretis_data.txt
        if md_items["status"] == "ACC":
            write_to_pathens(self, md_items["pnum_old"])

        self.sort_trajstate()
        self.config["current"]["traj_num"] = traj_num
        self.cworker = md_items["pin"]
        if self.printing():
            self.print_shooted(md_items, pn_news)
        # save for possible restart
        self.write_toml()

        return md_items

    def load_paths(self, paths):
        """Load paths."""
        size = self.n - 1
        # we add all the i+ paths.
        for i in range(size - 1):
            paths[i + 1].weights = calc_cv_vector(
                paths[i + 1],
                self.config["simulation"]["interfaces"],
                self.mc_moves,
                lambda_minus_one=self.config["simulation"]["tis_set"][
                    "lambda_minus_one"
                ],
                cap=self.cap,
            )
            self.add_traj(
                ens=i,
                traj=paths[i + 1],
                valid=paths[i + 1].weights,
                count=False,
            )
            pnum = paths[i + 1].path_number
            frac = self.config["current"]["frac"].get(
                str(pnum), np.zeros(size + 1)
            )
            self.traj_data[pnum] = {
                "ens_save_idx": i + 1,
                "max_op": paths[i + 1].ordermax,
                "min_op": paths[i + 1].ordermin,
                "length": paths[i + 1].length,
                "adress": paths[i + 1].adress,
                "weights": paths[i + 1].weights,
                "frac": np.array(frac, dtype="longdouble"),
            }
        # add minus path:
        paths[0].weights = (1.0,)
        pnum = paths[0].path_number
        self.add_traj(
            ens=-1, traj=paths[0], valid=paths[0].weights, count=False
        )
        frac = self.config["current"]["frac"].get(
            str(pnum), np.zeros(size + 1)
        )
        self.traj_data[pnum] = {
            "ens_save_idx": 0,
            "max_op": paths[0].ordermax,
            "min_op": paths[0].ordermin,
            "length": paths[0].length,
            "weights": paths[0].weights,
            "adress": paths[0].adress,
            "frac": np.array(frac, dtype="longdouble"),
        }

    def pattern_header(self):
        """Write pattern0 header."""
        if self.toinitiate == 0:
            restarted = self.config["current"].get("restarted_from")
            writemode = "a" if restarted else "w"
            with open(self.pattern_file, writemode) as fp:
                fp.write(
                    "# Worker\tMD_start [s]\t\twMD_start [s]\twMD_end",
                    +"[s]\tMD_end [s]\t Dask_end [s]",
                    +f"\tEnsembles\t{self.start_time}\n",
                )

    def initiate_ensembles(self):
        """Create all the ensemble dicts from the *toml config dict."""
        intfs = self.config["simulation"]["interfaces"]
        lambda_minus_one = self.config["simulation"]["tis_set"][
            "lambda_minus_one"
        ]
        ens_intfs = []

        # set intfs for [0-] and [0+]
        if lambda_minus_one is not False:
            ens_intfs.append(
                [lambda_minus_one, (lambda_minus_one + intfs[0]) / 2, intfs[0]]
            )
        else:
            ens_intfs.append([float("-inf"), intfs[0], intfs[0]])
        ens_intfs.append([intfs[0], intfs[0], intfs[-1]])

        # set interfaces and set detect for [1+], [2+], ...
        reactant, product = intfs[0], intfs[-1]
        for i in range(len(intfs) - 2):
            middle = intfs[i + 1]
            ens_intfs.append([reactant, middle, product])

        # create all path ensembles
        pensembles = {}
        for i, ens_intf in enumerate(ens_intfs):
            pensembles[i] = {
                "interfaces": tuple(ens_intf),
                "tis_set": self.config["simulation"]["tis_set"],
                "mc_move": self.config["simulation"]["shooting_moves"][i],
                "ens_name": f"{i:03d}",
                "start_cond": (
                    ["L", "R"]
                    if lambda_minus_one is not False and i == 0
                    else ("R" if i == 0 else "L")
                ),
            }

        self.ensembles = pensembles


def write_to_pathens(state, pn_archive):
    """Write data to infretis_data.txt."""
    traj_data = state.traj_data
    size = state.n

    with open(state.data_file, "a") as fp:
        for pn in pn_archive:
            string = ""
            string += f"\t{pn:3.0f}\t"
            string += f"{traj_data[pn]['length']:5.0f}" + "\t"
            string += f"{traj_data[pn]['max_op'][0]:8.5f}" + "\t"
            frac = []
            weight = []
            if len(traj_data[pn]["weights"]) == 1:
                f0 = traj_data[pn]["frac"][0]
                w0 = traj_data[pn]["weights"][0]
                frac.append("----" if f0 == 0.0 else str(f0))
                weight.append("----" if f0 == 0.0 else str(w0))
                frac += ["----"] * (size - 2)
                weight += ["----"] * (size - 2)
            else:
                frac.append("----")
                weight.append("----")
                for w0, f0 in zip(
                    traj_data[pn]["weights"][:-1], traj_data[pn]["frac"][1:-1]
                ):
                    frac.append("----" if f0 == 0.0 else str(f0))
                    weight.append("----" if f0 == 0.0 else str(w0))
            fp.write(
                string + "\t".join(frac) + "\t" + "\t".join(weight) + "\t\n"
            )
            traj_data.pop(pn)
