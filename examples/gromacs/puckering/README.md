## This is a work-in-progress exercise/ tutorial for a molecular modeling class.
TO DO
* step3 infretis
* step4 infretis
* step5 analysis
  * Trajectory visualization
  * plot 2d order parameter space
  * question about preferred mechanism from 2d order parameter plot
  * Visualize reactive trajectories?
  
# Motivation
See previous exercises. Something something rare events, path sampling simulations, ∞RETIS software, ...,

# Goals
The main goal of this exercise is to give you hands-on experience in performing a path simulation of a realistic system. A side quest is that you should be able to define your own molecular systems and learn how to generate the necessary force field files. During exercises 1 and 4 you learned to use Avogadro and GROMACS, and this will come in handy during this exercise.

# The system
We will study the [ring flip](https://en.wikipedia.org/wiki/Ring_flip) (often referred to as puckering) in some 6-ring-based system of your choosing.

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/puckering.gif" width="30%" height="30%">

This transition occurs very rarely at the molecular time scale, making it extremely tedious to study with standard molecular dynamics simulations. However, we would like to know exactly how often this transition occurs and the mechanism behind it. We can obtain this information by performing a path-sampling simulation, and in this exercise, you will carry out the whole modeling and analysis process from scratch.

The conformations of 6-rings are important in systems where they interact with other compounds in their environment. Examples include carbohydrates (6-ringed polymers) being broken down by enzymes at this very moment in your body.

The essential thing you need to know is that the conformational landscape of 6-rings can be classified into **C**hair, **H**alf-chair, **B**oat, **S**kew-boat, and **E**nvelope conformations. All these conformations are determined by the two angles $\theta$ and $\phi$, as illustrated in the figure below. There is a high energy barrier between the north pole and the equator, and again between the equator and the south pole. We will study the transition over the first barrier; starting at the north pole and ending at the equator.

<img src="http://enzyme13.bt.a.u-tokyo.ac.jp/CP/sugarconf.png" width="90%" height="90%">

## Questions
**1:** Given that the 6-ring in the animation above starts as $^4\text{C}_1$, what is the name of the ending structure? Hint: The super- and subscripts refer to which atoms are above and below the mean plane of the ring, respectively.

**2:** What is the initial value of the angle $\theta$, and what are the final values of the angles $\phi$ and $\theta$?

**3:** Can you suggest an order parameter for this transition?

# Installing the required packages
We first need to install the required programs to run this exercise. This includes a program that generates the parameters of a modern force field ([OpenFF 2.1](https://openforcefield.org/force-fields/force-fields/)) for your molecule and the ∞RETIS software developed at the theoretical chemistry group at NTNU.

Download and install mamba with the following commands (if you don't already have conda installed). Click the copy button on the box below and paste it into a terminal, and then do what is asked in the output on your screen.
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

```
Now close the terminal.

You should see `(base)` in the lower left of your terminal window after reopening if everything went successfully.

Then download and install the required python packages to run this exercise. Again copy-paste the code and do what is asked of you in the output.
```bash
mamba create --name molmod python==3.11 openff-toolkit-base ambertools rdkit pydantic
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
* **4:** We will perform the exercise from the directory `~/infretis/examples/gromacs/puckering/`. Get an overview of the folder structure and all the files we will be using by running
```bash
cd ~/infretis/examples/gromacs/puckering/
tree .

```

# Step 0: System definition and topology generation

Draw your favorite 6-ringed molecule in Avogadro in an $^4\text{C}_1$ conformation. Be sure to complete the valence of each atom.

The order parameter we will be using depends on the ring atoms, and we therefore need to identify the ring-atom indices. The atom indices can be accessed by checking the "Labels" box and then clicking "Atom Labels: Indices", as shown below:

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/labels.jpg" width="99%" height="99%">

Write down the atom indices in the following order:

* _idx0 idx1 idx2 idx3 idx4 idx5_
  
where _idx1_ and _idx4_ are the indices of the atoms 1 and 4, and we move clockwise from _idx0_ to _idx5_. In my case, the ordering is 

* 2 5 11 8 1 0

Optimize the structure and export it as `mol.sdf` in the `~/infretis/examples/gromacs/puckering/` folder (the .sdf format contains  coordinate, element, and bond order information).

Then run the following commands:

```bash
cd scripts
python generate-openff-topology.py ../mol.sdf
cd ../gromacs_input
gmx solvate -cs spc216.gro -cp mol.gro -p topol.top -o solv.gro
cd ..

```
## Questions
* **5:** What are the ordered ring atom indices for your system? Why do we care about these indices? 
* **6:** What do you think the commands above do?

# Step 1: Equilibration
The following commands perform an energy minimization and NVT and NPT equilibration runs.
```bash
cd step1_equilibration

# energy minimization
cd em
gmx grompp -f em.mdp -p ../../gromacs_input/topol.top -c ../../gromacs_input/solv.gro -o em.tpr
gmx mdrun -deffnm em -ntomp 2 -ntmpi 1 -pin on -v
cd -

```
```bash
# NVT equilibration
cd nvt
gmx grompp -f nvt.mdp -p ../../gromacs_input/topol.top -c ../em/em.gro -o nvt.tpr
gmx mdrun -deffnm nvt -ntomp 2 -ntmpi 1 -pin on -v
cd -

```
```bash
# NPT equlibration
cd npt
gmx grompp -f npt.mdp -p ../../gromacs_input/topol.top -c ../nvt/nvt.gro -t ../nvt/nvt.cpt -o npt.tpr
gmx mdrun -deffnm npt -ntomp 2 -ntmpi 1 -pin on -v

```
## Questions
* **7:** Has the temperature and density reached the expected values during the NPT equilibration?

# Step 2: MD run
Run a production run
```bash

# Production run
cd step2_md_run
gmx grompp -f md.mdp -p ../gromacs_input/topol.top -c ../step1_equilibration/npt/npt.gro -t ../step1_equilibration/npt/npt.cpt -o md.tpr
gmx mdrun -deffnm md -ntomp 2 -ntmpi 1 -pin on -v

```
We can process our trajectory files for visualization purposes. The following commands create a file `md-traj.xyz` that you can animate in Avogadro.
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

## Questions
* **8:** When visualizing the trajectory, do you see any spontaneous transitions? 
* **9:** What is the maximum order parameter value observed during the MD run?
* **10:** Given that the final product state of your molecule is defined by $\theta=90^{\circ}$, are you optimistic that you would observe a spontaneous transition using plain MD?

### PATH SIMULATION
