from scm.plams import *
init()
# You could also load the geometry from an xyz file:
# molecule = Molecule('path/my_molecule.xyz')
# or generate a molecule from SMILES:
# molecule = from_smiles('O')
molecule = Molecule()
molecule.add_atom(Atom(symbol='O', coords=(0,0,0)))
molecule.add_atom(Atom(symbol='H', coords=(1,0,0)))
molecule.add_atom(Atom(symbol='H', coords=(0,1,0)))

settings = Settings()
settings.input.ams.Task = 'GeometryOptimization'
settings.input.ams.Properties.NormalModes = 'Yes'
# settings.input.DFTB.Model = 'GFN1-xTB'
settings.input.ForceField.Type = 'UFF' # set this instead of DFTB if you do not have a DFTB license. You will then not be able to extract the HOMO and LUMO energies.

job = AMSJob(molecule=molecule, settings=settings, name='water_optimization')

print("-- input to the job --")
print(job.get_input())
print("-- end of input --")

job.run()

print(job.results.rkfpath(file='ams'))
print(job.results.rkfpath(file='engine'))
