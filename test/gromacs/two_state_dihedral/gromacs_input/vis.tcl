mol addfile /home/lukas/myinf/infretis/test/gromacs/infretis/gromacs_input/conf.gro
mol delrep 0 top
mol representation CPK 1.000000 0.300000 12.000000 12.000000
mol color Name
mol selection {all}
mol material Opaque
mol addrep top
label textsize 1.0
label add Atoms 0/0
label textoffset Atoms 0 {0.0 0.0}
label textformat Atoms 0 %R%d:%a
label add Atoms 0/1
label textoffset Atoms 1 {0.0 0.0}
label textformat Atoms 1 %R%d:%a
label add Atoms 0/2
label textoffset Atoms 2 {0.0 0.0}
label textformat Atoms 2 %R%d:%a
label add Atoms 0/3
label textoffset Atoms 3 {0.0 0.0}
label textformat Atoms 3 %R%d:%a
label add Bonds 0/0 0/1
label textoffset Bonds 0 {0.0 0.0}
label add Bonds 0/1 0/2
label textoffset Bonds 1 {0.0 0.0}
label add Bonds 0/2 0/3
label textoffset Bonds 2 {0.0 0.0}
label add Angles 0/0 0/1 0/2
label textoffset Angles 0 {0.0 0.0}
label add Angles 0/1 0/2 0/3
label textoffset Angles 1 {0.0 0.0}
label add Dihedrals 0/0 0/1 0/2 0/3
label textoffset Dihedrals 0 {0.0 0.0}
pbc box
