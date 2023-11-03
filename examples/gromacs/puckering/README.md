## This is a work-in-progress exercise/ tutorial for a molecular modeling class.
TO DO
* Remove the `cd` in commands, else its just copy-paste all the way
* step4 analysis/error analysis/mechanism
  * Trajectory visualization
  * plot 2d order parameter space
  * question about preferred mechanism from 2d order parameter plot
  * Visualize reactive trajectories?
* Update goals
* Fix motivation
  
# Motivation
See previous exercises. Something something rare events, path sampling simulations, ∞RETIS software, ...,

# Goals
The main goal of this exercise is to give you hands-on experience in performing a path simulation of a realistic system. A side quest is that you should be able to define your own molecular systems and learn how to generate the necessary force field files. During exercises 1 and 4 you learned to use Avogadro and GROMACS, and this will come in handy during this exercise.

# The system
We will study the [ring flip](https://en.wikipedia.org/wiki/Ring_flip) (often referred to as puckering) in some 6-ring-based system of your choosing.

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/puckering.gif" width="30%" height="30%">

This transition occurs very rarely at the molecular time scale, making it extremely tedious to study with standard molecular dynamics simulations. However, we would like to know exactly how often this transition occurs and the mechanism behind it. We can obtain this information by performing a path-sampling simulation, and in this exercise, you will carry out the whole modeling and analysis process from scratch.

The conformations of 6-rings are important in systems where they interact with other compounds in their environment. Examples include carbohydrates (6-ringed polymers) being broken down by enzymes at this very moment in your body.

The essential thing you need to know is that the conformational landscape of 6-rings can be classified into **C**hair, **H**alf-chair, **B**oat, **S**kew-boat, and **E**nvelope conformations. All these conformations are determined by the two angles $\theta$ and $\phi$, as illustrated in the figure below. There is a high energy barrier between the north pole and the equator, and again between the equator and the south pole.

<img src="http://enzyme13.bt.a.u-tokyo.ac.jp/CP/sugarconf.png" width="90%" height="90%">

 We will study the transition over the first barrier; _starting at the north pole and ending at any of the structures on the equator_. By the end of this exercise, you will be able to say exactly how often this transition happens, and which of the conformations at the equator your specific system prefers.
 
## Questions
**1:** Given that the 6-ring in the animation above starts as $^4\text{C}_1$, can you see that the ending structure is $^{3,O}B$? Hint: The super- and subscripts refer to which atoms are above and below the mean plane of the ring, respectively.

**2:** What is the initial value of the angle $\theta$, and what are the final values of the angles $\phi$ and $\theta$?

**3:** Can you suggest an order parameter for our transition?

# Installing the required packages
We first need to install the required programs to run this exercise. This includes a program that generates the parameters of a modern force field ([OpenFF 2.1](https://openforcefield.org/force-fields/force-fields/)) for your molecule and the ∞RETIS software developed at the theoretical chemistry group at NTNU.

Download and install mamba with the following commands (if you don't already have conda installed). Click the copy button on the box below and paste it into a terminal, and then do what is asked in the output on your screen (on Ubuntu, pressing down the mouse-wheel-button often works better for pasting than ctrl+v).
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

```
Now close the terminal.

You should see `(base)` in the lower left of your terminal window after reopening if everything went successfully.

Then download and install the required python packages to run this exercise. Again copy-paste the code and do what is asked of you in the output.
```bash
mamba create --name molmod python==3.11 openff-toolkit-base ambertools rdkit pydantic MDAnalysis
```
```bash
mamba activate molmod
mkdir software
cd software
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
python -m pip install -e .
cd -
git clone https://github.com/openforcefield/openff-models.git
cd openff-models
python -m pip install -e .
cd ~
git clone https://github.com/infretis/infretis.git
cd infretis
python -m pip install -e .
git checkout molmod_exercise5
cd examples/gromacs/puckering/
echo "All done! We will perform the exercise from this folder."

```

## Questions
* **4:** We will perform the exercise from the directory `~/infretis/examples/gromacs/puckering/`. Get an overview of the folder structure and all the files we will be using by navigating to that directory and running
```bash
ls *

```

# Step 0: System definition and topology generation

Draw your favorite 6-ringed molecule in Avogadro in an $^4\text{C}_1$ conformation. Be sure to complete the valence of each atom.

The order parameter we will be using depends on the ring atoms, and we therefore need to identify the ring-atom indices. The atom indices can be accessed by checking the "Labels" box and then clicking "Atom Labels: Indices", as shown below:

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/labels.jpg" width="99%" height="99%">

Write down the atom indices in the following order:

* _idx0 idx1 idx2 idx3 idx4 idx5_
  
where _idx1_ and _idx4_ are the indices of the atoms 1 and 4, and we move clockwise from _idx0_ to _idx5_. In my case, the ordering is 

* 2 5 11 8 1 0

**NOTE:** In some versions Avogadro starts counting at 0, in others at 1. If you don't see a 0-label in any of the atom labels, you need to subtract 1 from the above numbers. 

Optimize the structure and export it as `mol.sdf` in the `~/infretis/examples/gromacs/puckering/` folder (the .sdf format contains  coordinate, element, and bond order information). If you have problems here, you can use the `template.sdf` molecule. 

Check that you indeed are in the chair conformation with the given indices by using the `check-indices.py` script. Run `python check-indices.py -h` for usage information. You should obtain a  $\theta$ value close to $0^{\circ}$. 

Navigate to the `scripts` directory and run the following commands:

```bash
python generate-openff-topology.py ../mol.sdf
cd ../gromacs_input
gmx solvate -cs spc216.gro -cp mol.gro -p topol.top -o conf.g96
cd ..

```
## Questions
* **5:** What are the ordered ring atom indices for your system? Why do we care about these indices? 
* **6:** What do you think the commands above do?

# Step 1: Equilibration
Navigate to the `step1_equilibration` folder and get an overview of the directory structure. Perform an energy minimization and an NVT and NPT equilibration. Here are some commands to speed up the process. 
```bash
gmx grompp -f em.mdp -p ../../gromacs_input/topol.top -c ../../gromacs_input/conf.g96 -o em.tpr
gmx mdrun -deffnm em -ntomp 2 -ntmpi 1 -pin on -v
```
```bash
gmx grompp -f nvt.mdp -p ../../gromacs_input/topol.top -c ../em/em.gro -o nvt.tpr
gmx mdrun -deffnm nvt -ntomp 2 -ntmpi 1 -pin on -v

```
```bash
gmx grompp -f npt.mdp -p ../../gromacs_input/topol.top -c ../nvt/nvt.gro -t ../nvt/nvt.cpt -o npt.tpr
gmx mdrun -deffnm npt -ntomp 2 -ntmpi 1 -pin on -v -o

```
## Questions
* **7:** Has the temperature and density reached the expected values during the NPT equilibration? (Hint: Your system is mostly water)

# Step 2: MD run
Navigate to the `step2_md_run` folder and perform a production MD run. Remember to invoke `grompp` with the `-t` flag and give it the final state from the NPT simulation (see the NPT command for help).

We can process our trajectory files for visualization purposes. The following commands create a file `md-traj.xyz` that you can animate in Avogadro using the "Animation tool". 
```bash
# visualization
printf '1\n1\n' | gmx trjconv -f md.trr -pbc whole -center -o md-whole.xtc -s md.tpr
printf '1\n1\n' | gmx trjconv -f md-whole.xtc -fit rot+trans -s md.tpr -o md-traj.gro
obabel -igro md-traj.gro -oxyz -O md-traj.xyz
```

As you may have guessed by now, a good order parameter for the transition we want to study is the $\theta$ angle. To calculate the angle during the MD run, open `infretis.toml` and replace the indices with the ones you wrote down earlier. You can then recalculate the orderparameter using:

```bash
python ../scripts/recalculate-order.py -trr md.trr -toml infretis.toml -out md-order.txt

```
Plot the $\theta$ values (column 1) vs time (column 0) from the `md-order.txt` file. 

## Questions
* **8:** Do you see any interesting conformational changes when visualizing the trajectory?
* **9:** What is the maximum order parameter value observed during the MD run?
* **10:** Given that the product state of your molecule is defined by $\theta=90^{\circ}$, are you optimistic that you could observe a spontaneous transition during a plain MD simulation?

# Step 3: ∞RETIS
In this section, we will finally perform the path simulation. However, before we can do that, we need to provide the ∞RETIS program with a set of interfaces and an initial path in each of the path ensembles defined by the interfaces. 

We can cut out some paths for the lowest ensembles from the MD simulation, as these didn't reach high order parameter values. However, for an efficient simulation, we need to position interfaces far up the energy barrier, but these would be tedious to generate from plain MD simulations.

We can solve this problem in an iterative fashion by performing a couple of short ∞RETIS simulations. We start with the low-lying paths from the MD simulation and use these to start a short ∞RETIS simulation with low-lying interfaces. It is likely that we observe paths with higher order parameter values during this simulation. We can then use these paths as starting points in a second simulation with slightly higher interfaces. Continuing in this fashion effectively pushes the system up the energy barrier.

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/initial-paths.gif" width="45%" height="45%">

Navigate to the `step3_infretis` directory and modify the `infretis.toml` as follows:
* add your ring atom indices to the `[orderparameter]` section
* add two interfaces at 10.0 and 90.0 in the `interface = []` list. Remember a comma

We can cut out some paths with low order parameter values from the MD simulation by running:
```bash
python ../scripts/initial-path-from-md.py -trr ../step2_md_run/md.trr -toml infretis.toml -order ../step2_md_run/md-order.txt

```
You should now have created a `load` folder containing the paths and order parameter values for the two ensembles $[0^-]$ and $[0^+]$. Plot the order parameters for these two paths. You can use the `plot-order.py` script for this purpose. Run it with `-h` for help on usage.

If everything is in order, you should be able to run your first ∞RETIS simulation using:
```bash
infretisrun -i infretis.toml

```

We will now do the following iteratively:

* Plot the order parameter of all accepted paths (use the `plot-order.py` script on the `load/` folder). Stop if you observe a reactive path (one that crosses $\lambda_N=90^{\circ}$) and write down the printed path number(s).
* In this plot, identify the maximum order parameter of the *second* or *third* highest path (approximately). Add this value between $\lambda_0$ and $\lambda_N$ in your list of interfaces in `infretis.toml` (don't change the $\lambda_0=10^{\circ}$ and $\lambda_N=90^{\circ}$ interfaces). 
* Increase the number of `steps` in `infretis.toml` by 10.
* Rename the `load/` folder (so we don't overwrite it) to e.g. `run0`, or `run1`,`run2`, etc. if it exists
* Pick out some new initial paths for the next simulation from the previous simulation by using:

```bash
python ../scripts/initial-path-from-iretis.py -traj run0 -toml infretis.toml # generates a new load/ folder

```
* Run a new ∞RETIS simulation


After observing a reactive path, we assume that we have a reasonable set of interfaces and initial paths. Open the `restart.toml` file and change the number of `workers` to 4 and the number of `steps` to 1000. Fire off the simulation by invoking `infretis` with the `restart.toml` file. This will take some time (30-60 mins). 

While you wait, you can open another terminal and go on to the next step.

## Questions
* **11:** 
