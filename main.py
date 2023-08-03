from utils import *
from generator import Generator
from delpMetrics import ComputeMetrics
import argparse
import os
import sys
import csv
import glob
from os import listdir


def parser_defs(save_path, file_path):
    csv_file = open(save_path + '/defs.csv', 'w')
    writer = csv.writer(csv_file)
    for file_output in glob.glob(save_path + '*OUTPUT.json'):
        data = get_data_from_file(file_output)
        # write the header
        writer.writerow(['arg1','arg2','defeater'])
        for key, value in data.items():
            if len(value) != 0:
                for defeater in value:
                    # write the data
                    writer.writerow([key, defeater['defeat'], defeater['defeat']])
    csv_file.close()


class Test:

    def __init__(self):
        print("Starting...")

    def evaluate(self, path_to_generate):
        generator = Generator()
        # Get all parameters value
        #parameters = [[paraterms], [parameters]...]
        filenames = listdir(path_to_generate)
        test_parameters = np.genfromtxt(path_to_generate + '/' +filenames[0], delimiter=",")
        for idx, parameters in enumerate(test_parameters):
            parameters = parameters.tolist()
            local_params = {
                "KBBASE_SIZE": int(parameters[0]),
                "FACT_PROB": parameters[1],
                "NEG_PROB": 0.5,
                "DRULE_PROB": parameters[2],
                "MAX_RULESPERHEAD": int(parameters[3]),
                "MAX_BODYSIZE": int(parameters[4]),
                "MIN_ARGSLEVEL": int(parameters[5]),
                "LEVELS": int(parameters[6]),
                "RAMIFICATION": int(parameters[7]),
                "TREE_HEIGHT": int(parameters[8]),
                "INNER_PROB": 0.0,
                "N_PROGRAMS": 1,
                "PREF_CRITERION": "more_specific"
                }
            generator.generate(path_to_generate, idx, local_params)
        print("Complete")

    def test_generator(self, dataset_path):
        generator = Generator()
        params = get_data_from_file(dataset_path + 'parameters.json')
        generator.generate(dataset_path, params)
        print("Complete")

    def test_metrics_one(self, dir_path: str, delp_name: str, defs:bool) -> None:
        metrics = ComputeMetrics(dir_path, 'metric_one', dir_path, delp_name)
        metrics.show_setting()
        metrics.compute_one(defs)

    def test_metrics(self, dataset_path: str, defs:bool) -> None:
        if args.approx:
            metrics = ComputeMetrics(dataset_path, 'metrics_aprox', dataset_path, '')
        else:
            metrics = ComputeMetrics(dataset_path, 'metrics', dataset_path, '')
        metrics.show_setting()
        # List all program in the directory (but the parameters file)
        n_programs = os.listdir(dataset_path)
        metrics.compute_dataset(int((len(n_programs) - 1) / 2), defs, args.approx, args.perc)


test = Test()
parser = argparse.ArgumentParser(description='Test file for generate and metrics compute')
parser.add_argument('-load',
                    type=str,
                    help='The path for load dataset and save the results')
parser.add_argument('-all',
                    action='store_true',
                    help='To compute a dataset (generate dataset and compute the metrics)')
parser.add_argument('-gen',
                    action='store_true',
                    help='To only generate the programs')
parser.add_argument('-compute',
                    action='store_true',
                    help='To compute the metrics')
parser.add_argument('-approx',
                    action='store_true',
                    help='To compute an approximation value of metrics')
parser.add_argument('-perc',
                    type=int,
                    help='Percentage of literals to consult per level to approximate metrics values')
parser.add_argument('-one',
                    action='store_true',
                    help='To compute metrics for one program')
parser.add_argument('-p',
                    type=str,
                    help='DeLP program path')
# This is for generate a file with all defeater's
parser.add_argument('-defs',
                    action='store_true',
                    help='To print arguments-defeaters info'
                    )
parser.add_argument('-eval',
                    action='store_true',
                    help='To evaluate set of parameters')
#parser.add_argument('-defscript',
#                    action='store_true',
#                    help='To parse the info of the defeaters and arguments',
#                    )
#parser.add_argument('-fdefs',
#                    type=str,
#                    help='Path of the file with arguments and defeaters')

args = parser.parse_args()
input_path = args.load
program_path = args.p

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()
if args.eval:
    # Generate a set of programa and check is time
    test.evaluate(input_path)
if args.gen:
    # Generate a dataset
    test.test_generator(input_path)
if args.compute:
    # Compute metrics
    test.test_metrics(input_path, False)
if args.all:
    # Compute Dataset
    test.test_generator(input_path)
    test.test_metrics(input_path, args.defs)
elif args.one:
    # Compute one delp program
    test.test_metrics_one(input_path, program_path, args.defs)
# if args.defscript:
#     test.parser_defs(input_path, args.fdefs)
