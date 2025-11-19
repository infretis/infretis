from typing import Dict, List

from pydantic import BaseModel, Field, model_validator


class RunnerConfig(BaseModel):
    workers: int
    wmdrun: List[str] = Field(default_factory=list)


class TisSetConfig(BaseModel):
    maxlength: int
    allowmaxlength: bool
    zero_momentum: bool
    n_jumps: int
    interface_cap: float
    quantis: bool
    lambda_minus_one: bool
    accept_all: bool


class SimulationConfig(BaseModel):
    interfaces: List[float]
    steps: int
    seed: int
    load_dir: str
    shooting_moves: List[str]
    ensemble_engines: List[List[str]]
    tis_set: TisSetConfig


class EngineConfig(BaseModel):
    class_: str = Field(alias="class")
    engine: str
    timestep: float
    gmx_format: str
    input_path: str
    gmx: str
    subcycles: int
    temperature: float
    maxwarn: int


class OrderParameterConfig(BaseModel):
    class_: str = Field(alias="class")
    module: str
    config: str


class OutputConfig(BaseModel):
    data_dir: str
    screen: int
    pattern: int
    delete_old: bool
    keep_traj_fnames: List[str]
    data_file: str


class RngInnerState(BaseModel):
    state: int
    inc: int


class RngState(BaseModel):
    bit_generator: str
    has_uint32: int
    uinteger: int
    state: RngInnerState


class CurrentConfig(BaseModel):
    traj_num: int
    cstep: int
    active: List[int]
    locked: List[List[List[int | str]]]
    size: int
    frac: Dict[str, List[str]]
    rng_state: RngState


class FullConfig(BaseModel):
    runner: RunnerConfig
    simulation: SimulationConfig
    engine: EngineConfig
    orderparameter: OrderParameterConfig
    output: OutputConfig
    current: CurrentConfig

    @model_validator(mode="after")
    def validate_all(self) -> "FullConfig":
        sim = self.simulation
        eng = self.engine
        run = self.runner

        # shooting_moves must match number of interface intervals (minus one or equal depending on RETIS)
        if len(sim.shooting_moves) != len(sim.interfaces):
            raise ValueError(
                f"shooting_moves ({len(sim.shooting_moves)}) must match interfaces ({len(sim.interfaces)})"
            )

        # ensemble_engines must match interfaces or interface windows
        if len(sim.ensemble_engines) != len(sim.interfaces):
            raise ValueError(
                f"ensemble_engines ({len(sim.ensemble_engines)}) must match interfaces ({len(sim.interfaces)})"
            )

        # Worker count must equal number of wmdrun commands
        if len(run.wmdrun) != run.workers:
            raise ValueError(
                f"[runner.workers={run.workers}] but wmdrun has {len(run.wmdrun)} entries"
            )

        # Engine consistency rules (example)
        if eng.class_ == "gromacs":
            # Must specify a gmx binary
            if not eng.gmx:
                raise ValueError("GROMACS engine requires 'gmx' to be set")
            # Format required
            if eng.gmx_format not in {"gro", "g96", "pdb", "tpr"}:
                raise ValueError(
                    f"gmx_format '{eng.gmx_format}' is invalid for GROMACS"
                )

        # Temperature sanity check
        if eng.temperature <= 0:
            raise ValueError("temperature must be positive")

        # Subcycles ≥ 1
        if eng.subcycles < 1:
            raise ValueError("subcycles must be >= 1")

        # Order parameter file must be non-empty string
        if not isinstance(self.orderparameter.config, str):
            raise ValueError("orderparameter.config must be a string")

        # RNG state sanity
        if self.current.rng_state.has_uint32 not in (0, 1):
            raise ValueError("rng_state.has_uint32 must be 0 or 1")

        # Everything OK
        return self
