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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)

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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item2}')
                self.assertTrue(istrue)

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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item3}')
                self.assertTrue(istrue)

            # cd to previous, need to do this to delete tempdir
            os.chdir(file_path)
        os.chdir(curr_path)

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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)

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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item2}')
                self.assertTrue(istrue)

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
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item3}')
                self.assertTrue(istrue)

            # cd to previous, need to do this to delete tempdir
            os.chdir(file_path)
        os.chdir(curr_path)

    def test_pick_lock_sh(self):
        # get current path
        curr_path = pathlib.Path.cwd()

        # get path and cd into current file
        file_path = pathlib.Path(__file__).parent
        os.chdir(file_path)

        # here we unzip a "crashed" simulation (we finish step 10
        # but did not recieve step 11 path back. so we restart
        # from last restart.toml file.

        with tempfile.TemporaryDirectory(dir='./') as tempdir:
            # cd to tempdir
            os.chdir(tempdir)
            # copy files from template folder
            folder = 'data'
            shutil.copy(f'../{folder}/sh-crash.zip', './')
            shutil.unpack_archive('sh-crash.zip', './')
            os.system("infretisrun -i restart.toml >| out.txt")
            os.system("sed '53d' restart.toml >| restart0.toml")

            items0 = ['infretis_data.txt', 'restart0.toml']
            items1 = ['sh20steps.txt', 'sh20steps.toml']
            for item0, item1 in zip(items0, items1):
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)

            os.chdir(file_path)
        os.chdir(curr_path)

    def test_pick_lock_wf(self):
        # get current path
        curr_path = pathlib.Path.cwd()

        # here we unzip a "crashed" simulation (we finish step 10
        # but did not recieve step 11 path back. so we restart
        # from last restart.toml file.

        # get path and cd into current file
        file_path = pathlib.Path(__file__).parent
        os.chdir(file_path)

        with tempfile.TemporaryDirectory(dir='./') as tempdir:
            # cd to tempdir
            os.chdir(tempdir)
            # copy files from template folder
            folder = 'data'
            shutil.copy(f'../{folder}/wf-crash.zip', './')
            shutil.unpack_archive('wf-crash.zip', './')
            os.system("infretisrun -i restart.toml >| out.txt")
            os.system("sed '53d' restart.toml >| restart0.toml")

            items0 = ['infretis_data.txt', 'restart0.toml']
            items1 = ['wf20steps.txt', 'wf20steps.toml']
            for item0, item1 in zip(items0, items1):
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)

            os.chdir(file_path)
        os.chdir(curr_path)

# test what happens if we try to restart given restart.toml that does not have more steps.
if __name__ == '__main__':  
    unittest.main()
