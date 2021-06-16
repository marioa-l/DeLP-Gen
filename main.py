from utils import *
from generator import Generator
from delpMetrics import ComputeMetrics
import argparse
import os
import sys

# Delp datasets paths
#simple_test_path = "/mnt/dat/projects/programs/delp-gen/simple/"
#medium_test_path = "/mnt/dat/projects/programs/delp-gen/medium/"
#complex_test_path = "/mnt/dat/projects/programs/delp-gen/complex/"

# For test one delp program
#dlp_path = "/mnt/dat/projects/programs/delp-gen/simple/delp0.delp"


class Test:
    
    utils = Utils()
    

    def __init__(self):
        print("Testing...")


    def test_generator(self, dataset_path):
        generator = Generator()
        params = self.utils.get_data_from_file(dataset_path + 'parameters.json')
        generator.generate(dataset_path, params)
    

    def test_metrics_one(self, dir_path: str, delp_name: str) -> None:
        metrics = ComputeMetrics(dir_path, 'metric_one', dir_path, delp_name)
        metrics.show_setting()
        metrics.compute_one()


    def test_metrics(self, dataset_path: str) -> None:
        metrics = ComputeMetrics(dataset_path, 'metrics', dataset_path, '')
        metrics.show_setting()
        # List all program in the directory (but the parameters file)
        n_programs = os.listdir(dataset_path)
        metrics.compute_dataset(int((len(n_programs) - 1) / 2)) 
        #metrics.utils.write_metrics(result_path, metrics.build_path_result()) 

test = Test()
parser = argparse.ArgumentParser(description='Test file for generate and metrics compute')
parser.add_argument('-load',
                    type=str,
                    help='The path for load dataset and save the results')
parser.add_argument('-all',
                    action='store_true',
                    help='To compute a dataset')
parser.add_argument('-one',
                    action='store_true',
                    help='To compute metrics for one program')
parser.add_argument('-p',
                    type=str,
                    help='DeLP program path')
parser.add_argument('-gen',
                    action='store_true',
                    help='To only generate the programs')
args = parser.parse_args()

input_path = args.load
program_path = args.p

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()
if args.gen:
    test.test_generator(input_path)
if args.all:
    # Compute Dataset
    test.test_generator(input_path)
    test.test_metrics(input_path)
elif args.one:
    # Compute one delp program
    test.test_metrics_one(input_path, program_path)
