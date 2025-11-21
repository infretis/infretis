from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator


class TOMLConfigError(Exception):
    """Raised when there is an error in the .toml configuration."""

    pass


class InfinitConfig(BaseModel):
    pL: float = 0.3
    initial_conf: str = ""
    lamres: float = 0.01
    skip: float = 0.1
    cstep: int = -1
    steps_per_iter: List[int] = Field(default_factory=list)
    prev_Pcross: float = None


class RunnerConfig(BaseModel):
    """Workers: The number of MD simulations to be performed in parallel."""
    workers: int
    wmdrun: List[str] = Field(default_factory=list)


class BaseEngineConfig(BaseModel):
    class_: str = Field(alias="class")
    timestep: float
    subcycles: int
    temperature: float
    input_path: str = None

    model_config = {"extra": "allow"}

    def __getitem__(self, key):
        return getattr(self, key)


class TurtleMDEngine(BaseEngineConfig):
    class_: Literal["turtlemd"] = Field(alias="class")
    engine: str = "turtlemd"
    boltzmann: float = 1.0


class GromacsEngine(BaseEngineConfig):
    class_: Literal["gromacs"] = Field(alias="class")
    engine: str
    gmx_format: str
    gmx: str
    maxwarn: int = 0


class LammpsEngine(BaseEngineConfig):
    class_: Literal["lammps"] = Field(alias="class")
    engine: str
    lammps_bin: str


EngineType = Union[GromacsEngine, LammpsEngine, TurtleMDEngine]
EngineAdapter = TypeAdapter(EngineType)


class TisSetConfig(BaseModel):
    maxlength: int
    allowmaxlength: bool
    zero_momentum: bool = False
    n_jumps: int = 2
    quantis: bool = False
    accept_all: bool = False
    interface_cap: Optional[float] = None
    # need to cahnge bool to float
    lambda_minus_one: Optional[bool] = False


class SimulationConfig(BaseModel):
    ensemble_engines: Optional[List[List[str]]] = None
    interfaces: List[float]
    steps: int
    seed: int = 0
    load_dir: str
    shooting_moves: List[str]
    tis_set: TisSetConfig


class OrderParameterConfig(BaseModel):
    class_: str = Field(alias="class")
    model_config = {"extra": "allow"}

    def __getitem__(self, key):
        return getattr(self, key)


class OutputConfig(BaseModel):
    data_dir: str = "./"
    screen: int = 1
    delete_old: bool = False
    delete_old_all: bool = False
    keep_traj_fnames: List[str] = Field(default_factory=list)
    data_file: str = None
    keep_maxop_trajs: bool = False


class RngInnerState(BaseModel):
    state: int = None
    inc: int = None


class RngState(BaseModel):
    bit_generator: str = None
    has_uint32: int = None
    uinteger: int = None
    state: RngInnerState = Field(default_factory=RngInnerState)


class CurrentConfig(BaseModel):
    cstep: int = 0
    tsubcycles: int = 0
    traj_num: int = None
    size: int = None
    restarted_from: int = -1
    maxop: float = -float("inf")
    frac: Dict[str, List[str]] = Field(default_factory=dict)
    active: List[int] = Field(default_factory=list)
    locked: List[List[List[int | str]]] = Field(default_factory=list)
    rng_state: RngState = Field(default_factory=RngState)
    wsubcycles: List[int] = Field(default_factory=list)


class FullConfig(BaseModel):
    infinit: Optional[InfinitConfig] = None
    runner: RunnerConfig
    simulation: SimulationConfig
    engine: Dict[str, EngineType] | EngineType
    orderparameter: OrderParameterConfig
    output: OutputConfig
    current: CurrentConfig = Field(default_factory=CurrentConfig)

    # allow additional dics like "engine0", "engine1" ...
    model_config = {"extra": "allow"}

    # allow dict-like behaviour as well, e.g. FullConfig[key]
    # instead of only FullConfig.key
    def __getitem__(self, key):
        return getattr(self, key)

    @model_validator(mode="after")
    def validate_all(self) -> "FullConfig":
        # shorten variables for easier testing
        sim = self.simulation
        ts = sim.tis_set
        lms = ts.lambda_minus_one
        intfs = sim.interfaces
        cap = ts.interface_cap
        n_ens = len(sim.interfaces)
        nshm = len(sim.shooting_moves)
        nintf = len(sim.interfaces)
        curr = self.current

        # INFINIT check
        ## if no infinit settings
        if self.infinit is None:
            self.infinit = InfinitConfig(**{})

        # SIMULATION check
        has_ens_engs = sim.ensemble_engines
        if sim.ensemble_engines is None:
            sim.ensemble_engines = [["engine"] for _ in intfs]

        if ts.quantis and has_ens_engs is None:
            sim.ensemble_engines[0] = ["engine0"]


        # ENGINE checks
        ## Convert extra `engineX` blocks into EngineType models.
        unique_engines = {"engine": self.engine}
        for key, value in list(self.model_extra.items()):
            if key.startswith("engine"):
                self.model_extra[key] = EngineAdapter.validate_python(value)
                unique_engines[key] = self.model_extra[key]


        # check all engine names exist
        ens_engs = sum(sim.ensemble_engines, [])
        for ens_eng in set(ens_engs):
            if ens_eng not in unique_engines:
                raise TOMLConfigError(f"Engine '{ens_eng}' not defined!")

        # gromacs check
        input_paths = []
        for key1, eng1 in unique_engines.items():
            if self[key1].class_ == "gromacs":
                input_paths.append(eng1.input_path)
        if len(set(input_paths)) > 1:
            raise TOMLConfigError(
                "Found differing engine settings with identic"
                + "al 'input_path'. This would overwrite the"
                + " settings of one of the engines in"
                + " 'infretis.mdp'!"
            )

        # SIMULATION checks
        # if lms is not None and lms >= sim.interfaces[0]:
        #     raise TOMLConfigError(
        #         "lambda_minus_one interface must be \
        #         less than the first interface!"
        #     )

        if n_ens < 2:
            raise TOMLConfigError("Define at least 2 interfaces!")

        if sorted(sim.interfaces) != sim.interfaces:
            raise TOMLConfigError("Your interfaces are not sorted!")

        if len(set(sim.interfaces)) != len(sim.interfaces):
            raise TOMLConfigError("Your interfaces contain duplicate values!")

        if nshm < nintf:
            raise ValueError(
                f"number of shooting_moves ({nshm}) \
                must more or equal to number of interfaces ({nintf})"
            )

        if n_ens > nshm:
            raise TOMLConfigError(
                f"N_interfaces {n_ens} > N_shooting_moves {nshm}!"
            )

        if cap and cap > intfs[-1]:
            raise TOMLConfigError(
                f"Interface_cap {cap} > interface[-1]={intfs[-1]}"
            )

        if cap and cap < intfs[0]:
            raise TOMLConfigError(
                f"Interface_cap {cap} < interface[-2]={intfs[-2]}"
            )

        # CURRENT check

        ## build current
        if curr.traj_num is None:
            curr.traj_num = n_ens
        if not curr.wsubcycles:
            curr.wsubcycles = [0 for _ in range(self.runner.workers)]
        if not curr.active:
            curr.active = list(range(n_ens))
        if not self.current.size:
            curr.size = len(self.simulation.interfaces)

        ## set "restarted_from"
        if curr.cstep > 0:
            curr.restarted_from = curr.cstep

        # check wsubcycles
        if curr.wsubcycles is None:
            list_of_zeros = [0 for _ in range(self.runner.workers)]
            curr.wsubcycles = list_of_zeros

        # check tsubcycles
        if self.current.tsubcycles is None:
            self.current.tsubcycles = sum(self.current.wsubcycles)

        # add more in case of increased worker number
        if len(curr.wsubcycles) < self.runner.workers:
            extra = self.runner.workers - len(curr.wsubcycles)
            curr.wsubcycles += [0] * extra

        if self.runner.workers > len(intfs) - 1:
            raise TOMLConfigError("Too many workers defined!")

        return self
