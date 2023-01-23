import argparse
from infretis.scheduler import scheduler

def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                    help=f'Location of infretis input file',
                    required=True)

    args_dict = vars(parser.parse_args())
    input_file = args_dict['input']
    scheduler(input_file)

