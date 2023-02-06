import os
import unittest
import filecmp
import logging
import tempfile
import shutil
import pathlib
import tomli
import tomli_w
from unittest.mock import patch
from io import StringIO
from infretis.scheduler import scheduler
from distutils.dir_util import copy_tree


class test_infretisrun(unittest.TestCase):

    def test_infretisrun_1worker(self):
        # get path of where we run coverage unittest
        curr_path = pathlib.Path.cwd()

        # we collect istrue and check at the end to let tempdir
        # close without a possible AssertionError interuption
        true_list = []

        # get path and cd into current file
        file_path = pathlib.Path(__file__).parent
        os.chdir(file_path)

        with tempfile.TemporaryDirectory(dir='./') as tempdir:
            # cd to tempdir
            os.chdir(tempdir)
            # copy files from template folder
            folder = 'test_internal_1'
            shutil.copy(f'../{folder}/infretis.toml', './')
            shutil.copy(f'../{folder}/initial.xyz', './')
            shutil.copy(f'../{folder}/retis.rst', './')
            os.mkdir('trajs')
            copy_tree(f'../{folder}/trajs', './trajs')
            for ens in range(8):
                copy_tree(f'../{folder}/trajs/{ens}', f'../{folder}/trajs/e{ens}')
            
            # run standard simulation start command
            os.system("infretisrun -i infretis.toml >| out.txt")
            
            # compare files
            items1 = ['infretis_data.txt', 'restart.toml']
            for item in items1: 
                istrue = filecmp.cmp(f'./{item}', f'../{folder}/{item}')
                true_list.append(istrue)

            # edit restart.toml by increasing steps from 25 to 50
            with open('restart.toml', mode="rb") as f:
                config = tomli.load(f)
                config['simulation']['steps'] = 50
            with open("./restart.toml", "wb") as f:
                tomli_w.dump(config, f)       
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >| out.txt")

            # compare files
            items2 = ['infretis_data_restart.txt', 'restart_restart.toml']
            for item1, item2 in zip(items1, items2): 
                istrue = filecmp.cmp(f'./{item1}', f'../{folder}/{item2}')
                true_list.append(istrue)

            # cd to previous, need to do this to delete tempdir
            # os.chdir(file_path)
        os.chdir(curr_path)

        # finally check the istrues
        for istrue, item in zip(true_list, items1 + items2):
            if not istrue:
                print('fail!', item)
            self.assertTrue(istrue)


if __name__ == '__main__':  
    unittest.main()
