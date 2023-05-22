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

FILES = ['infretis.toml', 'gromacs.py', 'orderp.py', 'retis.rst']

class test_infretisrun(unittest.TestCase):

#     def test_infretisrun_1worker_wf_long(self):
#         # get path of where we run coverage unittest
#         curr_path = pathlib.Path.cwd()
# 
#         # get path and cd into current file
#         file_path = pathlib.Path(__file__).parent
#         os.chdir(file_path)
# 
#         with tempfile.TemporaryDirectory(dir='./') as tempdir:
#             # cd to tempdir
#             os.chdir(tempdir)
#             # copy files from template folder
#             folder = 'data'
#             for fil in FILES:
#                 shutil.copy(f'../{folder}/{fil}', './')
#             os.mkdir('trajs')
#             copy_tree(f'../{folder}/trajs', './trajs')
#             for ens in range(3):
#                 copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
# 
#             # run standard simulation start command
#             os.system("infretisrun -i infretis.toml >| out.txt")
#             # insert "restarted-from" line
#             # os.system("sed -i '55i restarted-from = 40' restart.toml")
# 
#             # compare files
#             items0 = ['infretis_data.txt', 'restart.toml']
#             items1 = ['wf20steps.txt', 'wf20steps.toml']
#             for item0, item1 in zip(items0, items1):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item1}')
#                 self.assertTrue(istrue)
#             os.chdir(file_path)
#         os.chdir(curr_path)

    def test_infretisrun_1worker_wf(self):
        # get path of where we run coverage unittest
        curr_path = pathlib.Path.cwd()

        # get path and cd into current file
        file_path = pathlib.Path(__file__).parent
        os.chdir(file_path)

        with tempfile.TemporaryDirectory(dir='./') as tempdir:
            # cd to tempdir
            os.chdir(tempdir)
            # copy files from template folder
            folder = 'data'
            for fil in FILES:
                shutil.copy(f'../{folder}/{fil}', './')
            os.mkdir('trajs')
            copy_tree(f'../{folder}/trajs', './trajs')
            for ens in range(3):
                copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
            
            # run standard simulation start command
            os.system("sed -i -e '16 s/20/5/' infretis.toml >> out.txt")
            os.system("infretisrun -i infretis.toml >| out.txt")
            
            # compare files
            items0 = ['infretis_data.txt', 'restart.toml']
            items1 = ['wf5steps.txt', 'wf5steps.toml']
            for item0, item1 in zip(items0, items1): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item1}')
                self.assertTrue(istrue)

            # edit restart.toml by increasing steps from 5 to 10
            os.system("sed -i -e '24 s/5/10/' restart.toml >> out.txt")
            
            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")
            # os.system("sed '55d' restart.toml >| restart0.toml")
            items0[1] = 'restart.toml'

            # compare files
            items2 = ['wf10steps.txt', 'wf10steps.toml']
            for item0, item2 in zip(items0, items2): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item2}')
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item2}')
                self.assertTrue(istrue)

            # # edit restart.toml by increasing steps from 10 to 15
            os.system("sed -i -e '24 s/10/15/' restart.toml >> out.txt")

            # restart standard simulation start command
            os.system("infretisrun -i restart.toml >> out.txt")
            os.system("sed '55d' restart.toml >| restart0.toml")

            # compare files
            items3 = ['wf15steps.txt', 'wf15steps.toml']
            for item0, item3 in zip(items0, items3): 
                istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item3}')
                if not istrue:
                    print(f'./{item0}', f'../{folder}/{item3}')
                self.assertTrue(istrue)

            # # edit restart.toml by increasing steps from 15 to 20
            # os.system("sed -i -e '24 s/15/20/' restart.toml >> out.txt")

            # # restart standard simulation start command
            # os.system("infretisrun -i restart.toml >> out.txt")
            # os.system("sed '55d' restart.toml >| restart0.toml")

            # # compare files
            # items3 = ['wf20steps.txt', 'wf20steps.toml']
            # for item0, item3 in zip(items0, items3): 
            #     istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item3}')
            #     if not istrue:
            #         print(f'./{item0}', f'../{folder}/{item3}')
            #     self.assertTrue(istrue)

            # cd to previous, need to do this to delete tempdir
            os.chdir(file_path)
        os.chdir(curr_path)
# 
#     def test_pick_lock_wf1(self):
#         # get current path
#         curr_path = pathlib.Path.cwd()
# 
#         # here we unzip a "crashed" simulation (we finish step 10
#         # but did not recieve step 11 path back. so we restart
#         # from last restart.toml file.
# 
#         # get path and cd into current file
#         file_path = pathlib.Path(__file__).parent
#         os.chdir(file_path)
# 
#         with tempfile.TemporaryDirectory(dir='./') as tempdir:
#             # cd to tempdir
#             os.chdir(tempdir)
#             # copy files from template folder
#             folder = 'data'
#             for fil in FILES:
#                 shutil.copy(f'../{folder}/{fil}', './')
#             os.mkdir('trajs')
#             copy_tree(f'../{folder}/trajs', './trajs')
#             shutil.copy(f'../{folder}/wfcrash1.zip', './')
#             shutil.unpack_archive('wfcrash1.zip', './')
#             for ens in range(3):
#                 copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
#                 shutil.move(f'p{ens}.restart', f'./trajs/{ens}/ensemble.restart')
#                 shutil.move(f'e{ens}.restart', f'./trajs/e{ens}/ensemble.restart')
# 
#             os.system("infretisrun -i restart.toml >| out.txt")
#             # remove "restarted-from" line
#             os.system("sed '55d' restart.toml >| restart0.toml")
# 
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items1 = ['wf10steps.txt', 'wf10steps.toml']
#             for item0, item1 in zip(items0, items1):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item1}')
#                 self.assertTrue(istrue)
# 
#             # # edit restart.toml by increasing steps from 10 to 15
#             with open('restart.toml', mode="rb") as f:
#                 config = tomli.load(f)
#                 config['simulation']['steps'] = 15
#             with open("./restart.toml", "wb") as f:
#                 tomli_w.dump(config, f)
# 
#             # restart standard simulation start command
#             os.system("infretisrun -i restart.toml >> out.txt")
#             # # running a zero step sim should not affect the following results
#             os.system("infretisrun -i restart.toml >> out.txt")
# 
#             os.system("sed '55d' restart.toml >| restart0.toml")
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items1 = ['wf15steps.txt', 'wf15steps.toml']
#             for item0, item1 in zip(items0, items1):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item1}')
#                 self.assertTrue(istrue)
# 
#             # edit restart.toml by increasing steps from 15 to 20
#             with open('restart.toml', mode="rb") as f:
#                 config = tomli.load(f)
#                 config['simulation']['steps'] = 20
#             with open("./restart.toml", "wb") as f:
#                 tomli_w.dump(config, f)
# 
#             # # restart standard simulation start command
#             os.system("infretisrun -i restart.toml >> out.txt")
#             os.system("sed -i -e '55 s/30/20/' restart.toml >> out.txt")
# 
#             # do some comp here?
# 
#             # compare files
#             os.system("sed '55d' restart.toml >| restart0.toml")
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items2 = ['wf20steps.txt', 'wf20steps.toml']
#             for item0, item2 in zip(items0, items2):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item2}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item2}')
#                 self.assertTrue(istrue)
#             os.chdir(file_path)
#         os.chdir(curr_path)
# 
#     def test_pick_lock_wf2(self):
#         # get current path
#         curr_path = pathlib.Path.cwd()
# 
#         # here we unzip a "crashed" simulation (we finish step 10
#         # but did not recieve step 11 path back. so we restart
#         # from last restart.toml file.
# 
#         # get path and cd into current file
#         file_path = pathlib.Path(__file__).parent
#         os.chdir(file_path)
# 
#         with tempfile.TemporaryDirectory(dir='./') as tempdir:
#             # cd to tempdir
#             os.chdir(tempdir)
#             # copy files from template folder
#             folder = 'data'
#             for fil in FILES:
#                 shutil.copy(f'../{folder}/{fil}', './')
#             os.mkdir('trajs')
#             copy_tree(f'../{folder}/trajs', './trajs')
#             shutil.copy(f'../{folder}/wfcrash2.zip', './')
#             shutil.unpack_archive('wfcrash2.zip', './')
#             for ens in range(3):
#                 copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
#                 shutil.move(f'p{ens}.restart', f'./trajs/{ens}/ensemble.restart')
#                 shutil.move(f'e{ens}.restart', f'./trajs/e{ens}/ensemble.restart')
# 
#             os.system("infretisrun -i restart.toml >| out.txt")
#             # remove "restarted-from" line
#             os.system("sed '55d' restart.toml >| restart0.toml")
# 
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items1 = ['wf10steps.txt', 'wf10steps.toml']
#             for item0, item1 in zip(items0, items1):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item1}')
#                 self.assertTrue(istrue)
# 
#             # # edit restart.toml by increasing steps from 10 to 15
#             with open('restart.toml', mode="rb") as f:
#                 config = tomli.load(f)
#                 config['simulation']['steps'] = 15
#             with open("./restart.toml", "wb") as f:
#                 tomli_w.dump(config, f)
# 
#             # restart standard simulation start command
#             os.system("infretisrun -i restart.toml >> out.txt")
#             # # running a zero step sim should not affect the following results
#             os.system("infretisrun -i restart.toml >> out.txt")
# 
#             os.system("sed '55d' restart.toml >| restart0.toml")
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items1 = ['wf15steps.txt', 'wf15steps.toml']
#             for item0, item1 in zip(items0, items1):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item1}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item1}')
#                 self.assertTrue(istrue)
# 
#             # edit restart.toml by increasing steps from 15 to 20
#             with open('restart.toml', mode="rb") as f:
#                 config = tomli.load(f)
#                 config['simulation']['steps'] = 20
#             with open("./restart.toml", "wb") as f:
#                 tomli_w.dump(config, f)
# 
#             # # restart standard simulation start command
#             os.system("infretisrun -i restart.toml >> out.txt")
#             os.system("sed -i -e '55 s/30/20/' restart.toml >> out.txt")
# 
#             # do some comp here?
# 
#             # compare files
#             os.system("sed '55d' restart.toml >| restart0.toml")
#             items0 = ['infretis_data.txt', 'restart0.toml']
#             items2 = ['wf20steps.txt', 'wf20steps.toml']
#             for item0, item2 in zip(items0, items2):
#                 istrue = filecmp.cmp(f'./{item0}', f'../{folder}/{item2}')
#                 if not istrue:
#                     print(f'./{item0}', f'../{folder}/{item2}')
#                 self.assertTrue(istrue)
#             os.chdir(file_path)
#         os.chdir(curr_path)
# 
#     def test_delete_old(self):
#         # get path of where we run coverage unittest
#         curr_path = pathlib.Path.cwd()
# 
#         # get path and cd into current file
#         file_path = pathlib.Path(__file__).parent
#         os.chdir(file_path)
# 
#         # recorded old data: 615423c2d39
#         live_trajs = [11, 10, 8]
#         record = {i: (i < 3 or i in live_trajs) for i in range(12)}
# 
#         with tempfile.TemporaryDirectory(dir='./') as tempdir:
#             # cd to tempdir
#             os.chdir(tempdir)
#             # copy files from template folder
#             folder = 'data'
#             for fil in FILES:
#                 shutil.copy(f'../{folder}/{fil}', './')
#             os.mkdir('trajs')
#             copy_tree(f'../{folder}/trajs', './trajs')
#             for ens in range(3):
#                 copy_tree(f'./trajs/{ens}', f'./trajs/e{ens}')
# 
#             # edit infretis.toml to add bm settings
#             with open('infretis.toml', mode="rb") as f:
#                 config = tomli.load(f)
#                 config['output']['screen'] = 1
#                 config['output']['delete_old'] = True
#             with open("./infretis.toml", "wb") as f:
#                 tomli_w.dump(config, f)
# 
#             # run standard simulation start command
#             os.system("infretisrun -i infretis.toml >| out.txt")
# 
#             collect_true = []
#             fitems = os.listdir('./trajs/')
#             live_trajs = [11, 10, 8]
#             record = {i: (i < 3 or i in live_trajs) for i in range(12)}
#             for i in range(12):
#                 if str(i) in fitems:
#                     fitems0 = os.listdir(f'./trajs/{i}/accepted')
#                     collect_true.append(record[i] == (len(fitems0)>0))
#             os.chdir(file_path)
#         os.chdir(curr_path)
#         self.assertTrue(len(collect_true)>0)
#         self.assertTrue(all(collect_true))
