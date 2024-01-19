<h1 align="center">
The Ring Flip Enigma:

Unveiling Molecular Secrets with Path Sampling
</h1>

# Motivation
The motivation for this assignment is to introduce students to software designed for conducting path sampling. The software employed here is an in-house-developed Python code for running &infin;RETIS that interfaces with Gromacs to perform essential molecular dynamics (MD) steps. Through this assignment, we aim to demonstrate the capability of studying a transition process that is nearly impossible to investigate using conventional brute-force MD methods due to its rare event nature. The algorithms and software utilized in this assignment are the result of very recent active developments within the research group of Theoretical Chemistry.

If path sampling and software development sound interesting to you, to the extent that you would like to study them in more detail, please don't hesitate to get in touch with Titus and Anders to explore potential master projects. You can contact them at titus.van.erp@ntnu.no and anders.lervik@ntnu.no.

# Goals
In this exercise, you'll journey into the heart of molecular mysteries. Your primary goal is to gain hands-on experience by simulating the [ring flip](https://en.wikipedia.org/wiki/Ring_flip), an intriguing phenomenon often referred to as puckering. This transition, a rare occurrence at the molecular timescale, has puzzled scientists for ages. With your newfound knowledge in path sampling, you now hold the key to understanding its enigmatic mechanisms. Your quest? To reveal the secrets hidden within the molecular world.

# The system
<p align="center">
<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/puckering.gif" width="30%" height="30%">
</p>

You will be tasked with modeling a 6-ring-based molecule of your choice, fully immersed in an explicit solvent environment. A solvated system adds complexity and a multitude of challenging behaviors, making your modeling task all the more interesting. Armed with your skills in Avogadro and GROMACS from previous exercises, you're well-prepared for this quest.

This transition occurs very rarely at the molecular time scale, making it extremely challenging to study with standard molecular dynamics simulations. On the macroscale, these systems are awfully small and the transition happens exceedingly fast, making it almost impossible to study experimentally. Truly, this process remains hidden within the world of molecules! However, we would like to know exactly how often this transition occurs and the mechanism behind it. We can obtain this information by performing a path-sampling simulation, and in this exercise, you will carry out the whole modeling and analysis process from scratch.

6-rings play a vital role in the world of chemistry and biology, impacting systems as diverse as carbohydrates being broken down by enzymes within your very body. The physical and chemical properties of 6-rings are intimately linked to their shapes, and their conformational landscape is a puzzle to be unraveled, with **C**hair, **H**alf-chair, **B**oat, **S**kew-boat, and **E**nvelope conformations. The conformations of 6-rings can be projected onoto the surface of a sphere, where each conformer is uniquely specified by the angles $\theta$ and $\phi$.

<img src="http://enzyme13.bt.a.u-tokyo.ac.jp/CP/sugarconf.png" width="90%" height="90%">

These angles should not be viewed as regular angles between atoms, but rather as a coordinate transformation of the atoms that can be convienetily mapped onto the surface of a sphere [[1](https://doi.org/10.1021/ja00839a011)]. But the "hows" aren't important right now. The essential thing you need to know for now is that there is a high energy barrier between the north pole and the equator, and again between the equator and the south pole. We will study the transition over the first barrier; _starting at the north pole and ending at any of the structures on the equator_. By the end of this exercise, you will be able to say exactly how often this transition happens, and the most likely ending structures on the equator for your specific system.

## Questions
**1:** Given that the 6-ring in the animation above starts as $^4\text{C}_1$, can you see that the ending structure is $^{3,O}B$? Hint: The super- and subscripts refer to which atoms are above and below the mean plane of the ring, respectively.

**2:** What is the initial value of the angle $\theta$, and what are the final values of the angles $\phi$ and $\theta$?

**3:** Can you suggest an order parameter for this transition?

# Installing the required packages
We first need to install the required programs to run this exercise. This includes a program that generates the parameters of a modern force field ([OpenFF 2.1](https://openforcefield.org/force-fields/force-fields/)) for your molecule and the ∞RETIS software developed at the theoretical chemistry group at NTNU.

Download and install mamba with the following commands (if you don't already have conda installed). Click the copy button on the box below and paste it into a terminal, and then do what is asked in the output on your screen (on Ubuntu, pressing down the mouse-wheel-button often works better for pasting than ctrl+V).
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

```
Now close the terminal.

You should see `(base)` in the left of your terminal window after reopening if everything went successfully.

Then download and install the required python packages to run this exercise. Again copy-paste the code and do what is asked of you in the output.
```bash
mamba create --name molmod python==3.11 openff-toolkit-base ambertools rdkit pydantic MDAnalysis tqdm
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
cd -
git clone https://github.com/infretis/inftools.git
python -m pip install -e .
cd ~
git clone https://github.com/infretis/infretis.git
cd infretis
python -m pip install -e .
git checkout molmod_exercise5
cd examples/gromacs/puckering/
echo "All done! We will perform the exercise from this folder."

```

You should now see `(molmod)` in the left of your terminal. Whenever you open a new terminal, write `mamba activate molmod` to activate the required Python packages. Try it by opening a new terminal and running `python -c 'import MDAnalysis'` without activating the `molmod` environment. This should throw an error.

We will perform the exercise from the directory `~/infretis/examples/gromacs/puckering/`. Get an overview of the folder structure and all the files we will be using by navigating to that directory and running
```bash
ls *

```

# Step 0: System definition and topology generation

*This step is optional, and you can skip directly to the last code snippet in this section by renaming the `template.sdf` file into `mol.sdf`.*

If you don't already have Avogadro or you want to update, you can download the newest version for Linux using the command below, but remove your old Avogadro2-x86_64.AppImage file first if it exists.

```bash

wget https://github.com/OpenChemistry/avogadrolibs/releases/download/1.98.1/Avogadro2-x86_64.AppImage -P ~
chmod +x ~/Avogadro2-x86_64.AppImage
~/Avogadro2-x86_64.AppImage
```

Draw your favorite 6-ringed molecule in Avogadro in the $^4\text{C}_1$ conformation. Be sure to complete the valence of each atom. You can also add substituents to the ring, and if you feel daring, you can use a sugar.

The order parameter we will be using depends on the ring atoms, and we therefore need to identify the ring-atom indices. The atom indices can be accessed by checking the "Labels" box and then clicking "Atom Labels: Indices", as shown below:

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/labels.jpg" width="99%" height="99%">

Write down the atom indices in the following order:

* _idx0 idx1 idx2 idx3 idx4 idx5_

where _idx1_ and _idx4_ are the indices of the atoms 1 and 4, and we move clockwise from _idx0_ to _idx5_. In my case, the ordering is

* 2 5 11 8 1 0

**NOTE:** In some versions Avogadro starts counting at 0, in others at 1. If you don't see a 0-label in any of the atom labels, you need to subtract 1 from the above numbers.

Optimize the structure and export it as `mol.sdf` in the `~/infretis/examples/gromacs/puckering/` folder (the .sdf format contains  coordinate, element, and bond order information).

Check that you indeed are in the chair conformation with the given indices by using the `check_indices` script contained in our [inftools](/infretis/inftools/) program, which calculates the $\theta$ and $\phi$ values. Run
```bash
inft check_indices -sdf mol.sdf -idx 2 5 11 8 1 0
```
but replace the indices with the ones you found. You should obtain a  $\theta$ value between $0-15^{\circ}$.

Finally, run the following commands:

```bash
inft generate_openff_topology -sdf mol.sdf
cd gromacs_input
gmx solvate -cs spc216.gro -cp mol.gro -p topol.top -o conf.g96
cd ..

```
## Questions
* **4:** What are the ordered ring atom indices for your system? Why do we care about these indices?
* **5:** What do you think the commands in the last command block above do?

# Step 1: Equilibration
Navigate to the `step1_equilibration` folder and get an overview of the directory structure. Perform an energy minimization (EM) and an NVT and NPT equilibration in the provided directories. Here are some commands to speed up the process.
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
gmx mdrun -deffnm npt -ntomp 2 -ntmpi 1 -pin on -v

```
## Questions
* **6:** Has the temperature and density reached the expected values during the NPT equilibration? The properties are accessible using `gmx energy -f npt.edr`. (Hint: retaw yltsom si metsys ruoY. Hint2: The letters of the previous hint are reversed to avoid spoilers.)

# Step 2: MD run
We have now equilibrated our system, and are now going to perform a slightly longer MD run. Navigate to the `step2_md_run` folder and invoke `grompp` with the following command.
```bash
gmx grompp -f md.mdp -p ../gromacs_input/topol.top -c ../step1_equilibration/npt/npt.gro -t ../step1_equilibration/npt/npt.cpt -o md.tpr
```
Fire off `mdrun`. This should take a couple of minutes.

As you may have guessed by now, a good order parameter for the transition we want to study is the $\theta$ angle. To calculate the angle during the MD run, open `infretis.toml` and replace the indices with the ones you wrote down earlier. You can then recalculate the order parameter using:

```bash
inft recalculate_order -trr md.trr -toml infretis.toml -out md-order.txt

```
Plot the $\theta$ values (column 1) vs time (column 0) from the MD run using e.g. gnuplot.



If you want, you can also visualize the trajectories, which in many cases can be very insightful. The following commands remove the solvent molecules and create a file `md-traj.xyz` that you can animate in Avogadro using the "Animation tool".

```bash
# visualization without solvent
printf '1\n1\n' | gmx trjconv -f *.trr -pbc whole -center -o md-whole.xtc -s *.tpr
printf '1\n1\n' | gmx trjconv -f md-whole.xtc -fit rot+trans -s *.tpr -o md-traj.gro
obabel -igro md-traj.gro -oxyz -O md-traj.xyz
```

## Questions
* **7:** Do you see any interesting conformational changes when visualizing the trajectory?
* **8:** What is the maximum order parameter value observed during the MD run?
* **9:** Given that the product state of your molecule is defined by $\theta=90^{\circ}$, are you optimistic that you could observe a spontaneous transition during a plain MD simulation?

# Step 3: ∞RETIS
In this section, we will finally perform the path simulation. However, before we can do that, we need to provide the ∞RETIS program with a set of interfaces and an initial path in each of the path ensembles defined by the interfaces.

We can cut out some paths for the lowest ensembles from the MD simulation, as these didn't reach high order parameter values. However, for an efficient simulation, we need to position interfaces far up the energy barrier, but these would be tedious to generate from plain MD simulations.

We can iteratively solve this problem by performing a couple of short ∞RETIS simulations. We start with the low-lying paths from the MD simulation and use these to start a short ∞RETIS simulation with low-lying interfaces. We likely observe paths with higher order parameter values during this simulation. We can then use these paths as starting points in a second simulation with slightly higher interfaces. Continuing in this fashion effectively pushes the system up the energy barrier.

<img src="https://github.com/infretis/infretis/blob/molmod_exercise5/examples/gromacs/puckering/graphics/initial-paths.gif" width="45%" height="45%">

Navigate to the `step3_infretis` directory and modify the `infretis.toml` as follows:
* add your ring atom indices to the `[orderparameter]` section
* add two interfaces at 10.0 and 90.0 in the `interface = []` list. Remember a comma

We can cut out some paths with low order parameter values from the MD simulation by running:
```bash
inft initial_path_from_md -trr ../step2_md_run/md.trr -toml infretis.toml -order ../step2_md_run/md-order.txt

```
You should now have created a `load` folder containing the paths and order parameter values for the two ensembles $[0^-]$ and $[0^+]$. Plot the order parameters for these two paths. The script `plot_order` inftool-program does this for you, and we will use it repeatedly in the following steps.

```bash
inft plot_order -toml infretis.toml -traj load/

```

Now, if everything is in order, you should be able to run your first ∞RETIS simulation using:
```bash
infretisrun -i infretis.toml

```

We will now do the following iteratively (similar to the procedure in GIF above):

* Plot the order parameter of all accepted paths (use the `plot_order` tool on the `load/` folder). Do you see a reactive path?
* In this plot, identify the maximum order parameter of the highest path. The position of the next interface should be slightly below this value (e.g. $0.5^{\circ}$ below this maximum). If you didn't reach any higher values, double the number of `steps` in `restart.toml` and run `infretisrun -i restart.toml`. Then start again from the top with the plotting.
* Add this value to your list of interfaces in `infretis.toml` such that the values are sorted (don't change the $\lambda_0=10^{\circ}$ and $\lambda_N=90^{\circ}$ interfaces).
* Increase the number of `steps` in `infretis.toml` by 10.
* Rename the `load/` folder (so we don't overwrite it) to e.g. `run0`, or `run1`,`run2`, etc. if it exists
* Pick out some new initial paths for the next simulation from the previous simulation by using:

```bash
# NOTE: Replace runx with the name of the most recent run folder (run0 if this is your first run)
inft initial_path_from_iretis -traj runx -toml infretis.toml # generates a new load/ folder

```
* Run a new ∞RETIS simulation
* Go to the first step above and start over until you observe a reactive path (one that crosses $\theta=90^{\circ}$.

After observing a reactive path, we assume we have a reasonable set of interfaces and initial paths. Open the `restart.toml` file and change the number of `workers` to 4 and the number of `steps` to around 1000 to 3000. The following step may take some time (10 to 30 mins, depending on your hardware) and may generate 1 to 3 GB of data. Fire off the simulation by invoking `infretis` with the `restart.toml` file.

# Step 4: Analysis
The following analysis is performed within the `step3_infretis` folder.
## The transition mechanism
We can say something about the mechanism of the complete $^4\text{C}_1 \rightarrow ^1\text{C}_4$ transition of your molecule if we assume that the second barrier from the equator to the south pole is negligible. The final configuration of your reactive paths would then be the transition state of the whole $^4\text{C}_1 \rightarrow ^1\text{C}_4$ transition. This may be a crude approximation, and we could test it by running another path simulation.

Plot the $\phi$ vs. $\theta$ values of the trajectories using the `-xy 2 1` option in `plot_order`. Looking at the reactive trajectories, what is/are the preferred route(s) from $^4\text{C}_1$ to $^1\text{C}_4$?

If you want, you can confirm this by visualizing some of the reactive trajectories. The following command removes the solvent, centers your molecule, and reorders the trajectories output from ∞RETIS:

```bash
# replace 'nr' with the path number of some trajectory you want to visualize
nr=46
inft concatenate -path load/${nr} -tpr ../gromacs_input/topol.tpr -out path${nr}.xyz
```
Now you get a file `path${nr}.xyz`that you can visualize in Avogadro.

## The transition rate

When you approach a reasonable number of paths in your simulation you can start analyzing the output. The following script calculates the rate, along with some other properties such as the crossing probability and error estimates.

```bash
inft wham -toml infretis.toml -data infretis_data.txt
```
The running average of the rate is written to the `runav_rate.txt` file, with the value in the fourth column giving the best estimate for the rate.
You can plot it in `gnuplot`

```bash
# in gnuplot
set logscale y
plot 'runav_rate.txt' using 1:4 with linespoints title 'rate'
```

The last line/point in this file is the estimated transition rate using all paths. To get this into units of $\text{ps}^{-1}$, divide the rate by $c$ where

$$c=\text{subcycles}\cdot \text{timestep}$$

which is found in the `infretis.toml` file.

Other files you may want to plot are the `Pcross.txt` for the crossing probability as a function of $\theta$, the `runav_flux` and `runav_Pcross.txt` for the running average of the flux and the crossing probability, and the `errRATE.txt`, `errFLUX.txt`, and `errPtot.txt` files for estimates of the relative error in the corresponding properties.

## Questions
* **10:** What is/are the preferred transition structures of your molecule on the equator?
* **11:** What is the rate in units of $\text{ns}^{-1}$?
* **12:** What is the interpretation of the inverse of the rate (1/rate)? (Hint: noitisnart rep emit ni era stinu ehT).
* **13:** Inspect the last part of the `md.log` file from `step2_md_run` and write down the Performance in ns/day. This number says how many nanoseconds of simulation you generate in one day on your machine. From the value of the inverse rate, how many days would you have to wait to observe a single transition in a standard MD simulation?


# How to pass this exercise
Answer all of the 13 questions and show/discuss them with the teaching assistants.
