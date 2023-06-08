"""The functions to be used to run infretis via the terminal."""
import argparse
from infretis.setup import setup_config
from infretis.scheduler import scheduler
from infretis.tools.conv_inf_py import print_pathens
from infretis.tools.pattern import pattern


def infretisrun():
    """Read input and runs infretis."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Location of infretis input file',
                        required=True)

    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']
    config = setup_config(input_file)
    if config is None:
        return
    state,md_items,path = scheduler(config)
    return state,md_items,path


def infretisanalyze():
    """Convert infretis_data.txt to pyretis data."""
    parser = argparse.ArgumentParser()
    inp0 = 'infretis_data.txt'
    inp1 = 'pattern.txt'
    parser.add_argument('-i', '--input',
                        help='Location of infretis data file,'
                             + f' default is {inp0}',
                        required=False)
    parser.add_argument('-p', '--pattern',
                        help='Location of infretis pattern file,'
                             + f' default is {inp1}',
                        required=False)

    args_dict = vars(parser.parse_args())
    input_file = args_dict.get('input', inp0)
    pattern_file = args_dict.get('pattern', False)
    print_pathens(input_file)
    if pattern_file:
        pattern(pattern_file)
