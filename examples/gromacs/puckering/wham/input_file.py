import argparse
import tomli

parser = argparse.ArgumentParser()

parser.add_argument("-toml", help="The infretis.toml file for the simulation")
parser.add_argument("-data",help="The infretis_data.txt file")

args = parser.parse_args()

####################################### input parameters ########################################
ifile=args.data
with open(args.toml, "rb") as toml_file:
    toml_dict = tomli.load(toml_file)
lambda_interfaces = toml_dict["simulation"]["interfaces"]
nskip=0
lamres=0.05 #the resolution along the lambda-CV. Must be commnensurate with interfaces.
minblocks=5 #the minimal number of blocks in error analysis
CalcFE=False #Calculate Free Energy
################################################################################################## 
