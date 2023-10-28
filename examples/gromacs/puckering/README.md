## This is a work-in-progress exercise/ tutorial for a molecular modeling class.
TO DO
* plot 2d order parameter space
* question about preferred mechanism from 2d order parameter plot
# Motivation
See previous exercises. Something something rare events, path sampling simulations, ∞RETIS software, ...,

# Goals
The main goal of this exercise is to give you hands-on experience in performing a path simulation of a realistic system. A side quest is that you should be able to define your own molecular systems and learn how to generate the necessary force field files. During exercises 1 and 4 you learned to use Avogadro and GROMACS, and this will come in handy during this exercise.

# The system
We will study the [ring flip](https://en.wikipedia.org/wiki/Ring_flip) (often referred to as puckering) in some 6-ring-based system of your choosing.

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/puckering.gif" width="30%" height="30%">

This transition occurs very rarely at the molecular time scale, making it extremely tedious to study with standard molecular dynamics simulations. However, we would like to know how often this transition occurs and the mechanism behind it. We can obtain this information by performing a path-sampling simulation, and in this exercise, you will carry out the whole modeling and analysis process from scratch.

The conformations of 6-rings are important in systems where they interact with some other compounds in their environment. Examples include carbohydrates (6-ringed polymers) being broken down by enzymes at this very moment in your body. 

The essential thing you need to know is that the conformational landscape of 6-rings can be classified into **C**hair, **H**alf-chair, **B**oat, **S**kew-boat, and **E**nvelope conformations. All these conformations are determined by the two angles $\theta$ and $\phi$, as illustrated in the figure below.

<img src="http://enzyme13.bt.a.u-tokyo.ac.jp/CP/sugarconf.png" width="90%" height="90%">

There is a high energy barrier between the north pole and the equator, and again between the equator and the south pole. We will study the transition over the first barrier; starting at the north pole and ending at the equator.

## Questions
**1:** Given that the 6-ring in the animation above starts as $^4\text{C}_1$, what is the name of the ending structure? Hint: The super- and subscripts refer to which atoms are above and below the mean plane of the ring, respectively.

**2:** What is the initial value of the angle $\theta$, and what are the final values of the angles $\phi$ and $\theta$?

**3:** Can you suggest an order parameter for this transition? 

# Step 0: Installing the required packages
We first need to install the required programs to run this exercise. This includes a program that generates the parameters of a modern force field ([OpenFF 2.1](https://openforcefield.org/](https://openforcefield.org/force-fields/force-fields/))) for your molecule and the ∞RETIS software developed at the theoretical chemistry group at NTNU.

Download and install mamba with the following commands (if you don't already have conda installed). Click the copy button on the box below and paste it into a terminal, and then do what is asked in the output on your screen. 
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Now close the terminal. 

You should see "(base)" in the lower left of your terminal window after reopening if everything went successfully. 

Then download and install the required python packages to run this exercise. Again copy-paste the code and do what is asked of you in the output.
```bash
mamba create --name molmod python==3.11 openff-toolkit-base ambertools rdkit pydantic
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
cd examples/gromacs/puckering/ # we will perform the exercise within this folder
```

## Questions
There are no questions in this section.

# Step 1: System definition and topology generation

Draw your favorite 6-ringed molecule in Avogadro in an $^4\text{C}_1$ conformation. Be sure to complete the valence of each atom.

Optimize the structure and export it as "mol.sdf".

Due to the order parameter definition, the atoms should be numbered 0 1 2 3 4 5 6, as in the skeleton.pdb file. If you want to simulate a charged system you need to neutralize the system. Help for this is found during the exercise sessions. Be careful with placing bulky substituents into axial positions, as the ring may flip spontaneously during equilibration due to a preference for equitorial positions.

Copy files and solvate the system
```bash
cd scripts
python generate-openff-topology.py
cd ../gromacs_input
gmx solvate -cs spc216.gro -cp mol.gro -p topol.top -o solv.gro
cd ..
```
### Equilibration
Energy minimization and equilibration
```bash
cd equil

# energy minimization
cd em
gmx grompp -f ../../mdps/em.mdp -p ../../gromacs_input/topol.top -c ../../gromacs_input/solv.gro -o em.tpr
gmx mdrun -deffnm em -ntomp 2 -ntmpi 1 -pin on -v
cd -

# NVT equilibration
cd nvt
gmx grompp -f ../../mdps/nvt.mdp -p ../../gromacs_input/topol.top -c ../em/em.gro -o nvt.tpr
gmx mdrun -deffnm nvt -ntomp 2 -ntmpi 1 -pin on -v
cd -

# NPT equlibration
cd npt
gmx grompp -f ../../mdps/npt.mdp -p ../../gromacs_input/topol.top -c ../nvt/nvt.gro -t ../nvt/nvt.cpt -o npt.tpr
gmx mdrun -deffnm npt -ntomp 2 -ntmpi 1 -pin on -v
cd ../..
```
Run a production run
```bash

# Production run
cd md-run
gmx grompp -f ../mdps/md.mdp -p ../gromacs_input/topol.top -c ../equil/npt/npt.gro -t ../equil/npt/npt.cpt -o md.tpr
gmx mdrun -deffnm md -ntomp 2 -ntmpi 1 -pin on -v


# visualization
gmx trjconv -f md.trr -pbc whole -center -o md-whole.xtc
printf '1\natomnr 1 to 6\n' | gmx select -on -s md.tpr
printf '1\n0\n' | gmx trjconv -f md-traj.gro -fit rot+trans -s md.tpr -n index.ndx -o md-traj.gro

# setting up the order parameter
# puckering coordinates, oxygen index 0, anomeric carbon nr. 1, ..., clockwise
# what is the maximum order parameter value you observe
```
### PATH SIMULATION
