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

    def test_infretisrun_1worker_sh(self):
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
            folder = 'data'
            shutil.copy(f'../{folder}/infretis.toml', './')
            shutil.copy(f'../{folder}/initial.xyz', './')
            shutil.copy(f'../{folder}/sh.rst', './retis.rst')
            os.mkdir('trajs')
            copy_tree(f'../{folder}/trajs', './trajs')
            for ens in range(8):
                copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
            
            # run standard simulation start command
            os.system("infretisrun -i infretis.toml >| out.txt")
            
            # compare files
            items0 = ['infretis_data.txt', 'restart.toml']
            items1 = ['sh20steps.txt', 'sh20steps.toml']
            for item0, item1 in zip(items0, items1): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # edit restart.toml by increasing steps from 25 to 50
            with open('restart.toml', mode="rb") as f:
                config = tomli.load(f)
                config['simulation']['steps'] = 40
            with open("./restart.toml", "wb") as f:
                tomli_w.dump(config, f)       
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")

            # compare files
            items2 = ['sh40steps.txt', 'sh40steps.toml']
            for item0, item2 in zip(items0, items2): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item2}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # edit restart.toml by increasing steps from 50 to 100
            with open('restart.toml', mode="rb") as f:
                config = tomli.load(f)
                config['simulation']['steps'] = 60
            with open("./restart.toml", "wb") as f:
                tomli_w.dump(config, f)       
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")

            # compare files
            items3 = ['sh60steps.txt', 'sh60steps.toml']
            for item0, item3 in zip(items0, items3): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item3}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # cd to previous, need to do this to delete tempdir
            os.chdir(file_path)
        os.chdir(curr_path)

        # finally check the istrues
        for istrue, item in zip(true_list, items1 + items2 + items3):
            if not istrue:
                print('fail!', item)
            self.assertTrue(istrue)

    def test_infretisrun_1worker_wf(self):
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
            folder = 'data'
            shutil.copy(f'../{folder}/infretis.toml', './')
            shutil.copy(f'../{folder}/initial.xyz', './')
            shutil.copy(f'../{folder}/wf.rst', './retis.rst')
            os.mkdir('trajs')
            copy_tree(f'../{folder}/trajs', './trajs')
            for ens in range(8):
                copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
            
            # run standard simulation start command
            os.system("infretisrun -i infretis.toml >| out.txt")
            
            # compare files
            items0 = ['infretis_data.txt', 'restart.toml']
            items1 = ['wf20steps.txt', 'wf20steps.toml']
            for item0, item1 in zip(items0, items1): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # edit restart.toml by increasing steps from 25 to 50
            with open('restart.toml', mode="rb") as f:
                config = tomli.load(f)
                config['simulation']['steps'] = 40
            with open("./restart.toml", "wb") as f:
                tomli_w.dump(config, f)       
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")

            # compare files
            items2 = ['wf40steps.txt', 'wf40steps.toml']
            for item0, item2 in zip(items0, items2): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item2}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # edit restart.toml by increasing steps from 50 to 100
            with open('restart.toml', mode="rb") as f:
                config = tomli.load(f)
                config['simulation']['steps'] = 60
            with open("./restart.toml", "wb") as f:
                tomli_w.dump(config, f)       
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")

            # compare files
            items3 = ['wf60steps.txt', 'wf60steps.toml']
            for item0, item3 in zip(items0, items3): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item3}')
                self.assertTrue(istrue)
                true_list.append(istrue)

            # cd to previous, need to do this to delete tempdir
            os.chdir(file_path)
        os.chdir(curr_path)

        # finally check the istrues
        for istrue, item in zip(true_list, items1 + items2 + items3):
            if not istrue:
                print('fail!', item)
            self.assertTrue(istrue)

if __name__ == '__main__':  
    unittest.main()
