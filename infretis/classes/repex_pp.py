"""Defines the main REPEX class for path handling and permanent calc."""
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import tomli_w
from numpy.random import default_rng

from infretis.classes.repex import REPEX_state, spawn_rng
from infretis.classes.engines.factory import assign_engines
from infretis.classes.formatter import PathStorage
from infretis.core.core import make_dirs
from infretis.core.tis import calc_cv_vector

logger = logging.getLogger("main")  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())
DATE_FORMAT = "%Y.%m.%d %H:%M:%S"


class REPEX_state_pp(REPEX_state):
    """Define the REPPEX object."""


    def __init__(self, config, minus=False):
        """Initiate REPEX given confic dict from *toml file."""
        super().__init__(config, minus=True)


    @property
    def prob(self):
        """Calculate the P matrix. For REPPTIS there is no swap,

        so prob for a path in 1 ensemble is 100% and zero otherwise.

        For RETIS this initiates a permanent calc."""
        prop = np.identity(self.n)
        prop[-1,-1] = 0.
        self._last_prob = prop.copy()
        return prop*(1-np.array(self._locks))

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

        # 1) Pick ensemble number
        # 2) Take random number. If >: shoot. If <: swap.
        # 2b) swap: choose neighbour (left or right, 50%). --> Check if locked
        # The idea is written below, but should be adapted for _offset and
        # ghost ensemble.
        if self.rgen.random() < self.zeroswap:
            if ens == 0:  # swap_zero if we choose [0-]
                neighb = ens + 1
            elif ens == self.n - 2:  # last ensemble, only swap left
                neighb = ens - 1
            elif ens < self.n - 2 :
                neighb = ens + self.rgen.choice([-1,1])
            if self._locks[neighb] == 0:
                # if neighbour is not locked, do sth
                other_traj = self.pick_traj_ens(neighb)
                sorted_idx = np.argsort([neighb, ens])
                ens_nums = tuple(np.array([neighb, ens])[sorted_idx]-1)
                logger.info("Swapping %s and %s" % (neighb, ens))
                inp_trajs = tuple(np.array([other_traj, traj])[sorted_idx])


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

    def add_traj(self, ens, traj, valid, count=True, n=0):
        """Add traj to state and calculate P matrix."""
        valid = np.zeros(self.n)
        valid[ens+1] = 1.
        ens += self._offset
        assert valid[ens] != 0
        # invalidate last prob
        self._last_prob = None
        self._trajs[ens] = traj
        self.state[ens, :] = valid
        self.unlock(ens)

        # Calculate P matrix
        self.prob

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
                    "ptype": get_ptype(out_traj,
                                       *self.ensembles[ens_num + 1]['interfaces'])
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
                "ptype": get_ptype(paths[i + 1],
                                   *self.ensembles[i + 1]['interfaces'])
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
            "ptype": get_ptype(paths[0],
                               *self.ensembles[0]['interfaces'])
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
        ens_intfs.append([float("-inf"), intfs[0], intfs[0]])
        ens_intfs.append([intfs[0], intfs[0], intfs[1]])

        # set interfaces and set detect for [1+], [2+], ...
        # reactant, product = intfs[0], intfs[-1]
        for i in range(len(intfs) - 2):
            middle = intfs[i + 1]
            ens_intfs.append([intfs[i], intfs[i + 1], intfs[i + 2]])

        # create all path ensembles
        pensembles = {}
        for i, ens_intf in enumerate(ens_intfs):
            pensembles[i] = {
                "interfaces": tuple(ens_intf),
                "tis_set": self.config["simulation"]["tis_set"],
                "mc_move": self.config["simulation"]["shooting_moves"][i],
                "ens_name": f"{i:03d}",
                "must_cross_M": True if i > 0 else False,
                "start_cond": "R" if i == 0 else ["L", "R"],
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
            string += f"{traj_data[pn]['min_op'][0]:8.5f}" + '\t'
            string += f"{traj_data[pn]['ptype']}" + '\t'
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


def get_ptype(path, L, M, R):
    end_cond = 'L' if path.phasepoints[-1].order[0] < L else 'R'
    start_cond = 'R' if path.phasepoints[0].order[0] > R else 'L'
    return start_cond + 'M' + end_cond
