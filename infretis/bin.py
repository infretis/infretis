import argparse
from infretis.scheduler import scheduler
from infretis.conv_inf_py import print_pathens
from infretis.pattern import pattern


def infretisrun():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                    help=f'Location of infretis input file',
                    required=True)

    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']
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
    pattern_file = args_dict.get('pattern', inp1)
    print_pathens(input_file)
    pattern(pattern_file)
