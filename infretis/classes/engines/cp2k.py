"""A CP2K external MD integrator interface.

This module defines a class for using CP2K as an external engine.

Important classes defined here
------------------------------

CP2KEngine (:py:class:`.CP2KEngine`)
    A class responsible for interfacing CP2K.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
from pathlib import Path
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    Optional,
    Union,
    Tuple,
    List,
    Dict,
)

import numpy as np

from infretis.classes.engines.enginebase import EngineBase
from infretis.classes.engines.engineparts import (
    PERIODIC_TABLE,
    ReadAndProcessOnTheFly,
    box_matrix_to_list,
    box_vector_angles,
    convert_snapshot,
    look_for_input_files,
    read_xyz_file,
    write_xyz_trajectory,
    xyz_reader,
)

if TYPE_CHECKING:  # pragma: no cover
    from infretis.classes.formatter import FileIO
    from infretis.classes.path import Path as InfPath
    from infretis.classes.system import System

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.addHandler(logging.NullHandler())


OUTPUT_FILES = {
    "energy": "{}-1.ener",
    "restart": "{}-1.restart",
    "pos": "{}-pos-1.xyz",
    "vel": "{}-vel-1.xyz",
    "wfn": "{}-RESTART.wfn",
    "wfn-bak": "{}-RESTART.wfn.bak-",
}


class SectionNode:
    """A class representing a section in the CP2K input.

    Attributes:
        title: The title of the section
        parent: The parent section if this node represents a sub-section.
        settings: The setting(s) for this particular node.
        data: A section of settings if the node defines several settings.
        children: A set with the sub-sections of this section.
        level: An integer to remember how far down this node is.
            E.g., if the level is 2, this node is a sub-sub-section.
            This is used for printing.
        parents: A list representing the path from the node to the top
            section.
    """

    def __init__(
        self,
        title: str,
        parent: Optional[SectionNode],
        settings: List[str],
        data: Optional[Dict[str, str]] = None,
    ):
        """Initialise a node.

        Args:
            title: The title of the section.
            parent: The parent, if this section is a sub-section.
            settings: The settings defined in this section.
            data: A section of settings.
        """
        self.title = title
        self.parent = parent
        self.settings = settings
        if data:
            self.data = list(data)
        else:
            self.data = []
        self.children: set[SectionNode] = set()
        self.level = 0
        self.parents: Optional[List[str]] = (
            None  # TODO: Check if this can be []
        )

    def add_child(self, child: SectionNode) -> None:
        """Add a sub-section to the current section."""
        self.children.add(child)

    def get_all_parents(self) -> None:
        """Find the path to the top of the tree."""
        parents = [self.title]
        prev = self.parent
        while prev is not None:
            parents.append(prev.title)
            prev = prev.parent
        self.parents = parents[::-1]


def dfs_print(node: SectionNode, visited: set[SectionNode]) -> List[str]:
    """Walk through the nodes and print out text.

    Args:
        node: The object representing a C2PK section.
        visited: The set contains the nodes we have already visited.

    Returns:
        A list of strings, representing the CP2K input file.
    """
    out = []
    pre = " " * (2 * node.level)
    if not node.settings:
        out.append(f"{pre}&{node.title}")
    else:
        out.append(f'{pre}&{node.title} {" ".join(node.settings)}')
    for lines in node.data:
        out.append(f"{pre}  {lines}")
    visited.add(node)
    for child in node.children:
        if child not in visited:
            for lines in dfs_print(child, visited):
                out.append(lines)
    out.append(f"{pre}&END {node.title}")
    return out


def set_parents(listofnodes: List[SectionNode]) -> Dict[str, SectionNode]:
    """Set parents for all nodes."""
    node_ref: Dict[str, SectionNode] = {}

    def dfs_set(node: SectionNode, vis: set[SectionNode]) -> None:
        """DFS traverse the nodes."""
        if node.parents is None:
            node.get_all_parents()
            # This sets the parents, so the line below is fine
            par = "->".join(node.parents)  # type: ignore[arg-type]
            if par in node_ref:
                prev = node_ref.pop(par)
                par1 = f'{par}->{" ".join(prev.settings)}'
                par2 = f'{par}->{" ".join(node.settings)}'
                node_ref[par1] = prev
                node_ref[par2] = node
            else:
                node_ref[par] = node
        vis.add(node)
        for child in node.children:
            if child not in visited:
                dfs_set(child, vis)

    for nodes in listofnodes:
        visited: set[SectionNode] = set()
        dfs_set(nodes, visited)
    return node_ref


def read_cp2k_input(filename: Union[str, Path]) -> List[SectionNode]:
    """Read a CP2K input file.

    Args:
        filename: Path to the file to read.

    Returns:
        nodes: The root section nodes found in the file.
    """
    nodes: List[SectionNode] = []
    current_node: Optional[SectionNode] = None
    with open(filename, "r", encoding="utf-8") as infile:
        for lines in infile:
            lstrip = lines.strip()
            if not lstrip:
                # skip empty lines
                continue
            if lstrip.startswith("&"):
                strip = lstrip[1:].split()
                if lstrip[1:].lower().startswith("end"):
                    # current node is always set here, so the line below is
                    # fine:
                    current_node = current_node.parent  # type: ignore
                else:
                    if len(strip) > 1:
                        setts = strip[1:]
                    else:
                        setts = []
                    new_node = SectionNode(
                        strip[0].upper(), current_node, setts
                    )
                    if current_node is None:
                        nodes.append(new_node)
                    else:
                        new_node.level = current_node.level + 1
                        current_node.add_child(new_node)
                    current_node = new_node
            else:
                if current_node is not None:
                    current_node.data.append(lstrip)
    return nodes


def _add_node(
    target: str,
    settings: List[str],
    data: Dict[str, str],
    nodes: List[SectionNode],
    node_ref: Dict[str, SectionNode],
) -> None:
    """Just add a new node."""
    # check if this is a root node:
    root = target.find("->") == -1
    if root:
        new_node = SectionNode(target, None, settings, data=data)
        nodes.append(new_node)
    else:
        parents = target.split("->")
        title = parents[-1]
        par = "->".join(parents[:-1])
        if par not in node_ref:
            _add_node(par, [], {}, nodes, node_ref)
        parent = node_ref["->".join(parents[:-1])]
        new_node = SectionNode(title, parent, settings, data=data)
        new_node.level = parent.level + 1
        parent.add_child(new_node)
    node_ref[target] = new_node


def update_node(
    target: str,
    settings: List[str],
    data: Dict[str, str],
    node_ref: Dict[str, SectionNode],
    nodes: List[SectionNode],
    replace: bool = False,
) -> Optional[SectionNode]:
    """Update the given target node.

    If the node does not exist, it will be created.

    Args:
        target: The target node, on form root->section->subsection
        settings: The settings for the node.
        data: The data for the node.
        node_ref: A dictionary of all nodes in the tree.
        nodes: The root nodes.
        replace: If this is True and if the nodes have some data,
            the already existing data will be ignored. We also assume
            that the data is already formatted.
    """
    if target not in node_ref:  # add node
        # TODO: remove decommented try-except construction later
        # try:
        _add_node(target, settings, data, nodes, node_ref)
        # except KeyError:
        #    pass
        return None
    node = node_ref[target]
    new_data = []
    done = set()
    if not replace:
        for line in node.data:
            key = line.split()[0]
            if key in data:
                new_data.append(f"{key} {data[key]}")
                done.add(key)
            else:
                new_data.append(line)
        for key in data:
            if key in done:
                continue
            if data[key] is None:
                new_data.append(str(key))
            else:
                new_data.append(f"{key} {data[key]}")
        node.data = list(new_data)
    else:
        node.data = list(data)
    if settings is not None:
        if replace:
            node.settings = list(settings)
        else:
            node.settings += settings
    return node


def remove_node(
    target: str,
    node_ref: Dict[str, SectionNode],
    root_nodes: List[SectionNode],
) -> None:
    """Remove a node (and it's children) from the tree.

    Args:
        target: The target node, on form root->section->subcection.
        node_ref: A dict with all the nodes.
        root_nodes: The root nodes.
    """
    to_del = node_ref.pop(target, None)
    if to_del is None:
        pass
    else:
        # remove all it's children:
        visited: set[SectionNode] = set()
        nodes = [to_del]
        while nodes:
            node = nodes.pop()
            if node not in visited:
                visited.add(node)
                for i in node.children:
                    nodes.append(i)
        # remove the reference to this node from the parent
        parent = to_del.parent
        if parent is None:
            # This is a root node.
            root_nodes.remove(to_del)
        else:
            parent.children.remove(to_del)
        del to_del
        # TODO: Is the code below actually working?
        # It seems like key has the wrong type?
        for key in visited:
            _ = node_ref.pop(key, None)  # type: ignore[call-overload]


def update_cp2k_input(
    template: Union[str, Path],
    output: Union[str, Path],
    update: Optional[Dict[str, Any]] = None,
    remove: Optional[List[str]] = None,
) -> None:
    """Read a template input and create a new CP2K input.

    Args:
        template: The CP2K input file we use as a template.
        output: The CP2K input file we will create.
        update: The settings we will update.
        remove: The nodes we will remove.
    """
    nodes = read_cp2k_input(template)
    node_ref = set_parents(nodes)
    if update is not None:
        for target in update:
            value = update[target]
            settings = value.get("settings", [])
            replace = value.get("replace", False)
            data = value.get("data", {})
            update_node(
                target, settings, data, node_ref, nodes, replace=replace
            )
    if remove is not None:
        for nodei in remove:
            remove_node(nodei, node_ref, nodes)
    with open(output, "w", encoding="utf-8") as outf:
        for i, nodej in enumerate(nodes):
            vis: set[SectionNode] = set()
            if i > 0:
                outf.write("\n")
            outf.write("\n".join(dfs_print(nodej, vis)))
            outf.write("\n")


class BoxData(TypedDict, total=False):
    """Store information about CP2K simulation box."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    PERIODIC: str
    ABC: np.ndarray
    ALPHA_BETA_GAMMA: np.ndarray


def read_box_data(
    box_data: List[str],
) -> Tuple[Optional[np.ndarray], List[bool]]:
    """Read the box data.

    Args:
        box_data: The settings for the SUBSYS->CELL section.

    Returns:
        A tuple containing:
        - The box vectors, in the correct order as a 1D numpy.array
        - The periodic boundary setting for each dimension as
            a list of booleans.
    """
    vectors = ("A", "B", "C", "ABC", "ALPHA_BETA_GAMMA")
    strings = ("PERIODIC",)
    data: BoxData = {}
    for lines in box_data:
        for key in vectors + strings:
            if lines.startswith(f"{key} "):
                if key in strings:
                    data[key] = " ".join(  # type: ignore[literal-required]
                        lines.split()[1:]
                    )
                else:
                    data[key] = np.array(  # type: ignore[literal-required]
                        [float(i) for i in lines.split()[1:]]
                    )
    if all(key in data for key in ("A", "B", "C")):
        box_matrix = np.zeros((3, 3))
        box_matrix[:, 0] = data["A"]
        box_matrix[:, 1] = data["B"]
        box_matrix[:, 2] = data["C"]
        box = np.array(box_matrix_to_list(box_matrix))
    elif "ABC" in data and "ALPHA_BETA_GAMMA" in data:
        box_matrix = box_vector_angles(
            data["ABC"],
            data["ALPHA_BETA_GAMMA"][0],
            data["ALPHA_BETA_GAMMA"][1],
            data["ALPHA_BETA_GAMMA"][2],
        )
        box = np.array(box_matrix_to_list(box_matrix))
    elif "ABC" in data:
        box = data["ABC"]
    else:
        box = None

    periodic_setting = data.get("PERIODIC", "XYZ").upper()
    periodic = [axis in periodic_setting for axis in ["X", "Y", "Z"]]
    return box, periodic


def read_cp2k_energy(energy_file: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Read and return CP2K energies.

    Args:
        energy_file: The path to the input file to read.

    Returns:
        This dict contains the energy terms read from the CP2K energy file.
    """
    data = np.genfromtxt(energy_file, invalid_raise=False)
    energy = {}
    for i, key in ((1, "time"), (2, "ekin"), (3, "temp"), (4, "vpot")):
        try:
            energy[key] = data[:, i]
        except IndexError:
            logger.warning(
                "Could not read energy term %s from CP2K file %s",
                key,
                energy_file,
            )
    if "ekin" in energy and "vpot" in energy:
        energy["etot"] = energy["ekin"] + energy["vpot"]
    return energy


def read_cp2k_box(
    inputfile: Union[str, Path],
) -> Tuple[Optional[np.ndarray], List[bool]]:
    """Read the box from a CP2K file.

    Args:
        inputfile: Path to the file to read from.

    Returns:
        A tuple containing:
            - The box vectors.
            - A list with settings (True/False) for periodic boundary
              conditions. If the list entry is True, periodic
              boundaries should be applied to the corresponding dimension.
    """
    nodes = read_cp2k_input(inputfile)
    node_ref = set_parents(nodes)
    try:
        box, periodic = read_box_data(
            node_ref["FORCE_EVAL->SUBSYS->CELL"].data
        )
    except KeyError:
        logger.warning('No CELL found in CP2K file "%s"', inputfile)
        box = np.array(
            [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]
        )
        periodic = [True, True, True]
    return box, periodic


def guess_particle_mass(particle_no: int, particle_type: str) -> float:
    """Guess a particle mass from its type and convert to CP2K units.

    Args:
        particle_no: The particle number. This is only used
            for output to the logger.
        particle_type: The particle type as a string. This
            should be an element from the periodic table.

    Returns:
        The guessed particle mass as a float.
    """
    logger.info(
        (
            "Mass not specified for particle no. %i\n"
            'Will guess from particle type "%s"'
        ),
        particle_no,
        particle_type,
    )
    mass = PERIODIC_TABLE.get(particle_type, None)
    if mass is None:
        msg = (
            f"Could not infer mass for particle number {particle_no}"
            f" with element '{particle_type}' as its not in our periodic"
            "table."
        )
        raise ValueError(msg)
    else:
        particle_mass = 1822.8884858012982 * mass
        logger.info(
            "-> Using a mass of %f g/mol (%f in internal units)",
            mass,
            particle_mass,
        )
    return particle_mass


def kinetic_energy(
    vel: np.ndarray, mass: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Obtain the kinetic energy for given velocities and masses.

    Args:
        vel: The velocities
        mass: The masses as a column vector.

    Returns:
        A tuple containing:
            - The kinetic energy.
            - The kinetic energy tensor.
    """
    mom = vel * mass
    if len(mass) == 1:
        kin = 0.5 * np.outer(mom, vel)
    else:
        kin = 0.5 * np.einsum("ij,ik->jk", mom, vel)
    return kin.trace(), kin


def reset_momentum(vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Set the linear momentum of all particles to zero.

    Note:
        Velocities are modified in place **and** returned.

    Args:
        vel: The velocities of the particles in system.
        mass: The masses of the particles in the system.

    Returns:
        The modified velocities of the particles.
    """
    # TODO: ?avoid creating an extra dimension by indexing array with None?
    mom = np.sum(vel * mass, axis=0)
    vel -= mom / mass.sum()
    return vel


def write_for_run_vel(
    infile: Union[str, Path],
    outfile: Union[str, Path],
    timestep: float,
    nsteps: int,
    subcycles: int,
    posfile: str,
    vel: np.ndarray,
    name: str = "md_step",
    print_freq: Optional[int] = None,
) -> None:
    """Create input file to perform n steps.

    Note, a single step actually consists of a number of subcycles.
    But from InfRETIS' point of view, this is a single step.
    Further, we here assume that we start from a given xyz file and
    we also explicitly give the velocities here.

    Args:
        infile: Path to the input template to use.
        outfile: Path to the input file to create.
        timestep: The time step to use for the simulation.
        nsteps: The number of InfRETIS steps to perform.
        subcycles: The number of sub-cycles to perform.
        posfile: The (base)name for the input file to read positions from.
        vel: The velocities to set in the input.
        name: A name for the CP2K project.
        print_freq: How often we should print to the trajectory file.
    """
    if print_freq is None:
        print_freq = subcycles
    to_update: Dict[str, Any] = {
        "GLOBAL": {
            "data": [f"PROJECT {name}", "RUN_TYPE MD", "PRINT_LEVEL LOW"],
            "replace": True,
        },
        "MOTION->MD": {
            "data": {"STEPS": nsteps * subcycles, "TIMESTEP": timestep}
        },
        "MOTION->PRINT->RESTART": {
            "data": ["BACKUP_COPIES 0"],
            "replace": True,
        },
        "MOTION->PRINT->RESTART->EACH": {"data": {"MD": print_freq}},
        "MOTION->PRINT->VELOCITIES->EACH": {"data": {"MD": print_freq}},
        "MOTION->PRINT->TRAJECTORY->EACH": {"data": {"MD": print_freq}},
        "FORCE_EVAL->SUBSYS->TOPOLOGY": {
            "data": {"COORD_FILE_NAME": posfile, "COORD_FILE_FORMAT": "xyz"}
        },
        "FORCE_EVAL->SUBSYS->VELOCITY": {
            "data": [],
            "replace": True,
        },
        "FORCE_EVAL->DFT->SCF->PRINT->RESTART": {
            "data": ["BACKUP_COPIES 0"],
            "replace": True,
        },
    }
    for veli in vel:
        to_update["FORCE_EVAL->SUBSYS->VELOCITY"]["data"].append(
            f"{veli[0]} {veli[1]} {veli[2]}"
        )
    remove = ["EXT_RESTART", "FORCE_EVAL->SUBSYS->COORD"]
    update_cp2k_input(infile, outfile, update=to_update, remove=remove)


class CP2KEngine(EngineBase):
    """A class defining the interface to CP2K.

    Attributes:
        cp2k: The command for executing CP2K.
        input_path: The directory where the input files are stored.
        timestep: The time step used in the CP2K MD simulation.
        subcycles: The number of steps each CP2K run is composed of.
        extra_files: List of extra files which may be required to run CP2K.
        sleep: A time in seconds, used to wait for files to be ready.

    """

    def __init__(
        self,
        cp2k: str,
        input_path: str,
        timestep: float,
        subcycles: int,
        temperature: float,
        extra_files: Optional[List[str]] = None,
        exe_path: Union[str, Path] = Path(".").resolve(),
        sleep: float = 0.1,
    ):
        """Set up the CP2K MD engine.

        Args:
            cp2k: The CP2K executable.
            input_path: The path to the directory containing CP2K input files.
            timestep: The time step used in the CP2K simulation.
            subcycles: The number of steps each CP2K run is composed of.
            temperature: The temperature of the simulation.
            extra_files: List of extra files which may be required to run CP2K.
            exe_path: The path on which the engine is executed
            sleep: A time in seconds, used to wait for files to be ready.
        """
        super().__init__("CP2K external engine", timestep, subcycles)
        self.ext = "xyz"
        self.name = "cp2k"
        self.cp2k = shlex.split(cp2k)
        self.sleep = sleep
        logger.info("Command for execution of CP2K: %s", " ".join(self.cp2k))
        # Store input path:
        self.input_path = Path(exe_path) / input_path
        # Set the defaults input files:
        default_files = {
            "conf": f"initial.{self.ext}",
            "template": "cp2k.inp",
        }
        # Check the presence of the defaults input files or, if absent,
        # try to find then by extension.
        self.input_files = look_for_input_files(self.input_path, default_files)

        # add mass, temperature and unit information to engine
        # which is needed for velocity modification
        _, _, _, atoms = self._read_configuration(self.input_files["conf"])
        mass = []
        if atoms is not None:
            mass = [
                guess_particle_mass(i, name) for i, name in enumerate(atoms)
            ]
        self.mass = np.reshape(mass, (len(mass), 1))

        # Check that the temperature set in cp2k.inp and infretis.toml
        # are the same. Often one modifies one and forgets about the other
        # If no temperature is set in cp2k.inp its NVE with NVT shooting
        # and we proceed without any remarks (?)
        section = "MOTION->MD"
        nodes = read_cp2k_input(self.input_files["template"])
        node_ref = set_parents(nodes)
        md_settings = node_ref[section]
        for data in md_settings.data:
            if "temperature" in data.lower():
                cp2k_temp = float(data.split()[1])
                if cp2k_temp != temperature:
                    msg = (
                        f"The temperature in the cp2k input {cp2k_temp} !="
                        + f" {temperature} which is specified in the .toml. "
                        + "They should be the same since we generate "
                        + "velocities internally at the .toml temperature."
                    )
                    raise ValueError(msg)

        self.temperature = temperature
        self.kb = 3.16681534e-6  # hartree
        self._beta = 1 / (self.temperature * self.kb)

        # todo, these info can be processed by look_for_input_files using
        # the extra_files option.
        self.extra_files = []
        if extra_files is not None:
            for key in extra_files:
                fname = os.path.join(self.input_path, key)
                if not os.path.isfile(fname):
                    logger.critical(
                        'Extra CP2K input file "%s" not found!', fname
                    )
                else:
                    self.extra_files.append(fname)

    def _extract_frame(self, traj_file: str, idx: int, out_file: str) -> None:
        """Extract a frame from a trajectory file.

        This method is used by `self.dump_config` when we are
        dumping from a trajectory file. It is not used if we are
        dumping from a single config file.

        Args:
            traj_file: Path to the trajectory file to dump from.
            idx: The frame number we look for.
            out_file: Path to the file to dump to.
        """
        for i, snapshot in enumerate(read_xyz_file(traj_file)):
            if i == idx:
                box, xyz, vel, names = convert_snapshot(snapshot)
                if os.path.isfile(out_file):
                    logger.debug("CP2K will overwrite %s", out_file)
                write_xyz_trajectory(
                    out_file, xyz, vel, names, box, append=False
                )
                return
        logger.error(
            "CP2K could not extract index %i from %s!", idx, traj_file
        )

    def _propagate_from(
        self,
        name: str,
        path: InfPath,
        system: System,
        ens_set: Dict[str, Any],
        msg_file: FileIO,
        reverse: bool = False,
    ) -> Tuple[bool, str]:
        """
        Propagate with CP2K from the current system configuration.

        Here, we assume that this method is called after the propagate()
        has been called in the parent. The parent is then responsible
        for reversing the velocities and also for setting the initial
        state of the system.

        Note that the on-the-fly reading of data is currently only applicable
        for NVT simulations, as no box information is read from CP2K.

        Args:
            name: A name to use for the trajectory we are generating.
            path: This is the path we use to fill in phase-space points.
            system: The phase point to use as a starting point.
            ens_set: Settings for the ensembles.
            msg_file: An object we use for writing out messages that are useful
                for inspecting the status of the current propagation.
            reverse: If True, the system will be propagated backward in time.

        Returns:
            A tuple containing:
                - True if we generated an acceptable path, False, otherwise.
                - A text description of the status of the propagation. Can
                  be used to interpret the reason for the path not being
                  acceptable.
        """
        status = f"propagating with CP2K (reverse = {reverse})"
        interfaces = ens_set["interfaces"]
        logger.debug(status)
        success = False
        left, _, right = interfaces
        logger.debug("Adding input files for CP2K")
        # First, copy the required input files:
        self.add_input_files(self.exe_dir)
        # Get positions and velocities from the input file.
        initial_conf = system.config[0]
        xyz, vel, box, atoms = self._read_configuration(initial_conf)
        if box is None:
            box, _ = read_cp2k_box(self.input_files["template"])
        # Add CP2K input for N steps:
        run_input = os.path.join(self.exe_dir, "run.inp")
        write_for_run_vel(
            self.input_files["template"],
            run_input,
            self.timestep,
            path.maxlen,
            self.subcycles,
            os.path.basename(initial_conf),
            vel,
            name=name,
        )
        # Get the order parameter before the run:
        order = self.calculate_order(system, xyz=xyz, vel=vel, box=box)
        traj_file = os.path.join(self.exe_dir, f"{name}.{self.ext}")
        # Create a message file with some info about this run:
        msg_file.write(
            f'# Initial order parameter: {" ".join([str(i) for i in order])}'
        )
        msg_file.write(f"# Trajectory file is: {traj_file}")
        # Get CP2K output files:
        out_files = {}
        for key, val in OUTPUT_FILES.items():
            out_files[key] = os.path.join(self.exe_dir, val.format(name))
        restart_file = os.path.join(self.exe_dir, out_files["restart"])
        prestart_file = os.path.join(self.exe_dir, "previous.restart")
        wave_file = os.path.join(self.exe_dir, out_files["wfn"])
        pwave_file = os.path.join(self.exe_dir, "previous.wfn")

        # CP2K runner
        logger.debug("Executing CP2K %s: %s", name, "run.inp")
        cmd = self.cp2k + ["-i", "run.inp"]
        cwd = self.exe_dir
        inputs = None
        # from external.exe_command
        cmd2 = " ".join(cmd)
        logger.debug("Executing: %s", cmd2)

        out_name = "stdout.txt"
        err_name = "stderr.txt"

        if cwd:
            out_name = os.path.join(cwd, out_name)
            err_name = os.path.join(cwd, err_name)

        return_code = None
        cp2k_was_terminated = False

        with open(out_name, "wb") as fout, open(err_name, "wb") as ferr:
            exe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=fout,
                stderr=ferr,
                shell=False,
                cwd=cwd,
                preexec_fn=os.setsid,
            )
            # wait for trajectories to appear
            while not os.path.exists(out_files["pos"]) or not os.path.exists(
                out_files["vel"]
            ):
                sleep(self.sleep)
                if exe.poll() is not None:
                    logger.debug("CP2K execution stopped")
                    break

            # CP2K may have finished after last checking files
            # or it may have crashed without writing the files
            if exe.poll() is None or exe.returncode == 0:
                pos_reader = ReadAndProcessOnTheFly(
                    out_files["pos"], xyz_reader
                )
                vel_reader = ReadAndProcessOnTheFly(
                    out_files["vel"], xyz_reader
                )
                # start reading on the fly as CP2K is still running
                # if it stops, perform one more iteration to read
                # the remaining content in the files. Note that we assume here
                # that CP2K writes in blocks of frames, and never partially
                # finished frames.
                iterations_after_stop = 0
                step_nr = 0
                pos_traj = []
                vel_traj = []
                while exe.poll() is None or iterations_after_stop <= 1:
                    # we may still have some data in one of the trajectories
                    # so use += here
                    pos_traj += pos_reader.read_and_process_content()
                    vel_traj += vel_reader.read_and_process_content()
                    # loop over the frames that are ready
                    for frame in range(min(len(pos_traj), len(vel_traj))):
                        pos = pos_traj.pop(0)
                        vel = vel_traj.pop(0)
                        write_xyz_trajectory(traj_file, pos, vel, atoms, box)
                        # calculate order, check for crossings, etc
                        order = self.calculate_order(
                            system, xyz=pos, vel=vel, box=box
                        )
                        msg_file.write(
                            f'{step_nr} {" ".join([str(j) for j in order])}'
                        )
                        snapshot = {
                            "order": order,
                            "config": (traj_file, step_nr),
                            "vel_rev": reverse,
                        }
                        phase_point = self.snapshot_to_system(system, snapshot)
                        status, success, stop, add = self.add_to_path(
                            path, phase_point, left, right
                        )
                        if stop:
                            # process may have terminated since we last checked
                            if exe.poll() is None:
                                logger.debug("Terminating CP2K execution")
                                os.killpg(os.getpgid(exe.pid), signal.SIGTERM)
                                # wait for process to die, necessary for mpi
                                exe.wait(timeout=360)
                            logger.debug(
                                "CP2K propagation ended at %i. Reason: %s",
                                step_nr,
                                status,
                            )
                            # exit while loop without reading additional data
                            iterations_after_stop = 2
                            cp2k_was_terminated = True
                            break

                        step_nr += 1
                    sleep(self.sleep)
                    # if CP2K finished, we run one more loop
                    if exe.poll() is not None and iterations_after_stop <= 1:
                        iterations_after_stop += 1

            return_code = exe.returncode
            if return_code != 0 and not cp2k_was_terminated:
                logger.error(
                    "Execution of external program (%s) failed!",
                    self.description,
                )
                logger.error("Attempted command: %s", cmd2)
                logger.error("Execution directory: %s", cwd)
                if inputs is not None:
                    logger.error("Input to external program was: %s", inputs)
                logger.error(
                    "Return code from external program: %i", return_code
                )
                logger.error("STDOUT, see file: %s", out_name)
                logger.error("STDERR, see file: %s", err_name)
                msg = (
                    f"Execution of external program ({self.description}) "
                    f"failed with command:\n {cmd2}.\n"
                    f"Return code: {return_code}"
                )
                raise RuntimeError(msg)
        if (return_code is not None) and (
            return_code == 0 or cp2k_was_terminated
        ):
            self._removefile(out_name)
            self._removefile(err_name)

        msg_file.write("# Propagation done.")
        energy_file = out_files["energy"]
        msg_file.write(f"# Reading energies from: {energy_file}")
        energy = read_cp2k_energy(energy_file)
        end = (step_nr + 1) * self.subcycles
        ekin: np.ndarray = energy.get("ekin", np.array([]))
        vpot: np.ndarray = energy.get("vpot", np.array([]))
        path.update_energies(
            ekin[: end : self.subcycles], vpot[: end : self.subcycles]
        )
        for _, files in out_files.items():
            self._removefile(files)
        self._removefile(prestart_file)
        self._removefile(pwave_file)
        self._removefile(run_input)
        self._removefile(restart_file)
        self._removefile(wave_file)
        return success, status

    def add_input_files(self, dirname: str) -> None:
        """Add required CP2K input files to the given directory."""
        for files in self.extra_files:
            basename = os.path.basename(files)
            dest = os.path.join(dirname, basename)
            if not os.path.isfile(dest):
                logger.debug(
                    'Adding input file "%s" to "%s"', basename, dirname
                )
                self._copyfile(files, dest)

    @staticmethod
    def _read_configuration(
        filename: Union[str, Path],
    ) -> Tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray], Optional[List[str]]
    ]:
        """Read a CP2K output configuration from a file.

        Args:
            filename: Path to the file to read.

        Returns:
            A tuple containing:
                - The positions.
                - The velocities.
                - The box dimensions if we manage to read it.
                - The atom names found in the file.
        """
        for snapshot in read_xyz_file(filename):
            box, xyz, vel, names = convert_snapshot(snapshot)
            break  # Stop after the first snapshot.
        return xyz, vel, box, names

    def set_mdrun(self, md_items: Dict[str, Any]) -> None:
        """Set the execute directory."""
        # TODO: REMOVE OR RENAME?
        self.exe_dir = md_items["exe_dir"]

    def _reverse_velocities(self, filename: str, outfile: str) -> None:
        """Reverse velocity in a given snapshot.

        Args:
            filename: Path to the file containing the configuration to
                reverse velocities in.
            outfile: Path to file for storing the configuration with
                reversed velocities.
        """
        xyz, vel, box, names = self._read_configuration(filename)
        write_xyz_trajectory(
            outfile, xyz, -1.0 * vel, names, box, append=False
        )

    def modify_velocities(
        self, system: System, vel_settings: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Modify the velocities of all particles.

        Args:
            system: The system whose particle velocities are to be modified.
            vel_settings: A dict containing
                'zero_momentum': boolean, if true we reset the linear momentum
                  to zero after generating velocities internally.

        Returns:
            A tuple containing:
                - dek: The change in kinetic energy as a result of
                    the velocity modification.
                - kin_new: The new kinetic energy of the system.

        Note:
            CP2K removes the center of mass motion by default,
            so we set the momentum to zero by default here.
            This method does **not** take care of constraints.
        """
        mass = self.mass
        beta = self.beta
        pos = self.dump_frame(system)
        xyz, vel, box, atoms = self._read_configuration(pos)
        kin_old = kinetic_energy(vel, mass)[0]
        # TO DO: Is this check neccessary?
        if box is None:
            box, _ = read_cp2k_box(self.input_files["template"])
        vel, _ = self.draw_maxwellian_velocities(vel, mass, beta)
        # reset momentum by default in cp2k
        if vel_settings.get("zero_momentum", True):
            vel = reset_momentum(vel, mass)

        conf_out = os.path.join(self.exe_dir, f"genvel.{self.ext}")
        write_xyz_trajectory(conf_out, xyz, vel, atoms, box, append=False)
        kin_new = kinetic_energy(vel, mass)[0]
        system.config = (conf_out, 0)
        system.ekin = kin_new
        if kin_old == 0.0:
            dek = float("inf")
            logger.debug(
                "Kinetic energy not found for previous point."
                "\n(This happens when the initial configuration "
                "does not contain energies.)"
            )
        else:
            dek = kin_new - kin_old
        return dek, kin_new
