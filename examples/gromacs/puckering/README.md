## This is a work-in-progress exercise/ tutorial for a molecular modeling class.

### Installing required packages
Install mamba if you do not allready have conda
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Close and reopen the terminal. Then install the required packages to run this exercise
```bash
mamba create --name molmod python==3.11 openff-toolkit-base ambertools rdkit pydantic
mkdir software
cd software
git clone https://github.com/openforcefield/openff-interchange.git
cd openff-interchange
python -m pip install -e .
cd -
git clone https://github.com/openforcefield/openff-models.git
cd openff-models
python -m pip install -e .
cd -
git clone https://github.com/infretis/infretis.git
cd infretis
python -m pip install -e .
cd -
```

### Topolgy generation

build molecule in avogadro and export it with filename 'mol.sdf'. Remember to optimize the geometry before saving the molecule. If you feel daring, you can modify the substituents of the 6-ring, but do not change the indices of the ring atoms. Due to the order parameter defenition they should be numbered 0 1 2 3 4 5 6, as in the skeleton.pdb file. If you want to simulate a charged system you need to neutralize the system. Help for this is found during the exercise sessions. Be careful with placing bulky substituents into equitorial positions, as the ring may flip spontaneously during equilibration due to a preference for equitorial positions.

Copy files and solvate the system
```bash
cd script
python generate-openff-topology.py
cd ../gromacs_input
gmx solvate -cs spc216.gro -cp mol.gro -p topol.top -o gromacs_input/solv.gro
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
