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
    
    def test_metrics(self, result_path: str, dataset_path: str, program: str) -> None:
        metrics = ComputeMetrics(result_path, 'metrics', dataset_path, program)
        metrics.show_setting()
        metrics.compute_dataset(10)
        #metrics.utils.write_metrics(result_path, metrics.build_path_result()) 

test = Test()
parser = argparse.ArgumentParser(description='Test file for generate and metrics compute')
parser.add_argument('Path',
                    type=str,
                    help='The path for load dataset and save the results')

args = parser.parse_args()

input_path = args.Path

if not os.path.isdir(input_path):
    print('The path specified does not exist')
    sys.exit()

test.utils.print_info("Working on path: " + input_path)
test.test_generator(medium_test_path)
test.utils.print_info("Dataset created...")
test.test_metrics(medium_test_path, medium_test_path, '')
