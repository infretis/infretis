# infretis

1. Requires https://gitlab.com/danielzh/pyretis/-/tree/dz24/dask to be installed.
2. Run "python -m pip install -e ." in this root folder
3. pip install dask distributed tomli_w
4. run pyretisrun -i retis.rst with load and steps=0. to copy ensemble.restart files..
5. If you have external orderp.py file, need to specfiy in infretis.toml dask > files = ["orderp.py"].
6. The bash scripts in examples/internal/ and examples/gromacs/ explains how to run infretis.

Todo:
* inhouse path sampling code
	1. Classes & Functions (from pyretis&ops)
	2. Smart restart
	3. Toy external engine
	4. Reimplement subtrajectory MC moves
	5. Multiple runnable ways (pin, hpc, etc.)
