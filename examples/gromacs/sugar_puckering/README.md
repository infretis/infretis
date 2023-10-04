# This is a work-in-progress exercise/tutorial for a molecular modeling class.

# 		TOPOLOGY GENERATION
# if you don't have conda or miniconda
# you can install it on the computers in the computer room by using
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# create a new virtual environment
conda create --name iretis

# without acpype? OpenFF?
# install acpype (which includes ambertools using conda) to generate force field files
conda install -c conda-forge acpype

# build molecule in avogadro and export it with
# filename 'mol.pdb'. Remember to optimize the geometry
# before saving the molecule.
# If you feel daring, you can modify the substituents of the 6-ring, but
# do not change the indices of the ring atoms. Due to the order
# parameter defenition they should be numbered
# 0 1 2 3 4 5 6, as in the skeleton.pdb file.
# If you want to simulate a charged system you need to neutralize the system.
# Help for this is found during the exercise sessions.
# Be careful with placing bulky substituents into equitorial positions, as the ring may flip
# spontaneously during equilibration due to a preference for equitorial positions.
acpype -i mol.pdb -o gmx
cd gromacs_input
cp ../mol.acpype/mol_GMX.itp mol.itp
cp ../mol.acpype/mol_GMX.gro mol.gro

# ------------------ equil.sh start
# solvate the system
gmx editconf -f mol.gro -box 2.0 -c -bt cubic -o newbox.gro
gmx solvate -cs spc216.gro -cp newbox.gro -p topol.top -o solv.gro
cd ../equil

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
cd -
# ------------------ equil.sh end

# Production run
cd ../..

cd md-run
gmx grompp -f ../mdps/md.mdp -p ../gromacs_input/topol.top -c ../equil/npt/npt.gro -t ../equil/npt/npt.cpt -o md.tpr
gmx mdrun -deffnm md -ntomp 2 -ntmpi 1 -pin on -v


# visualization
gmx trjconf -f md.trr -pbc whole -center -o md-whole.xtc
printf '1\natomnr 1 to 6\n' | gmx select -on -s md.tpr
printf '1\n0\n' | gmx trjconv -f md-traj.gro -fit rot+trans -s md.tpr -n index.ndx -o md-traj.gro

# setting up the order parameter
# puckering coordinates, oxygen index 0, anomeric carbon nr. 1, ..., clockwise
# what is the maximum order parameter value you observe

#		PATH SIMULATION
# download and install infretis
git clone https://github.com/infretis/infretis.git
cd infretis
python -m pip install -e .
