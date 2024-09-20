<h1 align="center">
Water autoionization
</h1>

### TODO
* reduce load path size
* upload to infentory
* make scripts for plotting?
* make scripts for visualization
* fix lammpstrj processor
* fix xyz processor?

### Aloha 👋
In this session, we will study the autoionization of water using **path sampling**. The main outcomes of such a simulation allow us to

* calculate 🖥️ exactly how often a chemical reaction occurs
* and visualize 👀 how this chemical reaction actually happens


To achieve this you are going to perform the following steps:
* 1️⃣ Perform a molecular dynamics (MD) simulation using [LAMMPS](https://www.lammps.org/#nogo) to equilibrate your system
* 2️⃣ Perform a path sampling simulation on this system with &infin;RETIS + LAMMPS
* 3️⃣ See and learn how water dissociates at the molecular scale 🔎

### Step 0: Installation
Open a terminal.


If you don't already have conda or mamba:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b
```

Close and then re-open the terminal. You should now see **(base)** in the lower left of your screen.

Download python and lammps:

```bash
mamba create --name cosy_24 python==3.11 lammps
```

Install infretis, and the exercise files.
```bash
mamba activate cosy_24
python -m pip install git+https://github.com/infretis/infretis.git@cosy_24
python -m pip install git+https://github.com/infretis/inftools.git@main
git clone https://github.com/infretis/infentory.git cosy_workshop
cd cosy_workshop/water_dissociation/lammps/
echo ========== We will perform the exercise from this folder ===============
```

### Step 1: Equilibration with LAMMPS

### Step 2: Path sampling with &infin;RETIS

### Step 3: Analysis of the results


