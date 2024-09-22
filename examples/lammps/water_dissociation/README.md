<h1 align="center">
Water autoionization
</h1>

### TODO
* [x] reduce load path size
* [ ] upload to infentory
* [ ] make scripts for plotting?
* [x] make scripts for visualization
* [x] fix lammpstrj processor
* [x] fix xyz processor?
* [ ] also adapt puckering exercise for cosy; exactly the same as here but with gromacs. Only production md run and path sampling, then visualize
* [ ] move puckering and water dissociation README.md to infentory
* [ ] fix installation after moving to infentory
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

### Step 1: MD with LAMMPS
Familiarize yourself with the files in the directory `lammps_input/`. Can you explain what these files contain?

Now, change to the `step1_md_run` directory and modify `lammp.input` to run an MD simulation at 300K for around 1 picosecond with a 0.5 fs timestep. We want to analyze some of the output, so write output with reasonable frequency.

LAMMPS can be run with the command `lmp -i lammps.input`.

Does the system reach the desired temperature?

Animate the trajectory by opening the .dump file in Avogadro.

```bash
# Download Avogadro
wget https://github.com/OpenChemistry/avogadrolibs/releases/download/1.99.0/Avogadro2-x86_64.AppImage ~/
chmod +x ~/Avogadro2-x86_64.AppImage
# Run Avogadro
~/Avogadro2-x86_64.AppImage
```
Use the `animation tool` and check the `Dynamic bonding?` checkbox. Do you see anything interesting? You can rotate and zoom with the `navigation tool`

Investigate the value of the order parameter. Values up to around 1.5 tell that we have only water present (the largest O-H bond length of all water molecules). Values greater than 1.5 tell us we have one OH- and one H3O+ present (the shortest distance between the OH- oxygen and H3O+ hydrogen).

```bash
inft recalculate_order -toml infretis.toml -traj md_run.dump -format 'lammpsdump'
```

[insert image here]

### Step 2: Path sampling with &infin;RETIS + LAMMPS

### Step 3: Analysis of the results
Interested reader: This is a reproduction of the work in cite pnas?

```bash
inft trjcat -out traj.pdb -traj trash/reactive/traj.txt -topology test_reaxff_water/initial.xyz -format lammpsdump -centersel "index 78"
inft trjcat -out traj.pdb -traj traj.txt -topology ../../../gromacs_input/topol.tpr -centersel "element C" -selection "index 0 to 15"
