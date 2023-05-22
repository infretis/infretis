import argparse
import tomli
from infretis.core.scheduler import scheduler
from infretis.tools.conv_inf_py import print_pathens
from infretis.tools.pattern import pattern


def infretisrun():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                    help=f'Location of infretis input file',
                    required=True)

    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']

    # check here temporary
    with open(input_file, mode="rb") as f:
        config = tomli.load(f)
        if 'current' in config:
            current = config['current']
            restart = current.get('restarted_from', False)
            cstep = current.get('cstep', False)
            if True in set((restart, cstep)) and cstep == restart:
                print('current step and total steps are equal so we exit ',
                      'without doing anything.')
                return
    scheduler(input_file)


def infretisanalyze():
    parser = argparse.ArgumentParser()
    inp0 = 'infretis_data.txt'
    inp1 = 'pattern.txt'
    parser.add_argument('-i', '--input',
                    help=f'Location of infretis data file, default is {inp0}',
                    required=False)
    parser.add_argument('-p', '--pattern',
                    help=f'Location of infretis pattern file, default is {inp1}',
                    required=False)

    args_dict = vars(parser.parse_args())
    input_file = args_dict.get('input', inp0)
    pattern_file = args_dict.get('pattern', False)
    print_pathens(input_file)
    if pattern_file:
        pattern(pattern_file)
