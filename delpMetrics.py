import json
import sys
import time
from progress.spinner import Spinner
from subprocess import STDOUT, check_output
from utils import *
import re

class ComputeMetrics:
    
    def __init__(self,
                 path_file_results: str,
                 file_results_name: str,
                 path_dataset: str,
                 path_delp: str) -> None:
        """
        The constructor for the experiment class 
        Args:
            -path_file_results: The path for save results files
            -file_results_name: The name of the result file
        """
        self.path_file_results = path_file_results
        self.file_results_name = file_results_name
        self.path_dataset = path_dataset
        self.path_delp = path_delp
        self.aux_height = []
        self.utils = Utils()
        self.times = []
        #self.patter_presumption = \[\([a-zA-Z]*\-\<[t|T]rue\)\]


    def show_setting(self) -> None:
        """
        Show experiment settings
        """
        self.utils.print_info("Output path: " + self.path_file_results)
        self.utils.print_info("Result file: " + self.file_results_name)
        self.utils.print_info("Dataset: " + self.path_dataset)
        self.utils.print_info("Delp program: " + self.path_delp)


    def build_path_result(self) -> str:
        """
        Build the path where the results will be saved
        """
        return self.path_file_results + self.file_results_name + '.json'


    def query_literal_solver(self, literal: str) -> None:
        """
        Call to delp solver to query for one literal
        Args:
            -literal: The literal to consult
        """
        delpProgram = self.path_delp
        cmd = ['./globalCore', 'file', delpProgram, 'answ', literal]
        
        try:
            output = check_output(cmd, stderr=STDOUT, timeout=60). \
                decode(sys.stdout.encoding)
            result = output
            return result
        except Exception as e:
            print(e)
            exit()
            return "Error"


    def query_delp_solver(self) -> json:
        """
        Call to delp solver to get all answers for the delp program
        in self.path_delp
        """
        delpProgram = self.path_delp
        cmd = ['./globalCore', 'file', delpProgram, 'all']
        try:
            output = check_output(cmd, stderr=STDOUT, timeout=60). \
                decode(sys.stdout.encoding)
            result = json.loads(output)
            return result
        except Exception as e:
            print(e)
            return "Error"


    def count_lines(self, root: int, lines: list, level=0) -> int:
        """
        Count the number of lines of a dialectical tree with root <root>
        Args:
            -root: The id of the root argument
            -lines: List of all defeat relationships to build all arg lines
                    [[arg, defeater, id_arg, id_defeater],...]
        """
        childs = [defeaters[3] for defeaters in lines if defeaters[2] == root]
        if len(childs) == 0:
            # is leaf
            self.aux_height.append(level)
            return 1  # Line, Heigth
        line = 0
        for child in childs:
            line += self.count_lines(child, lines, level + 1)
        return line
        

    def analyze_results(self, result):
        n_arguments = 0.0
        n_defeaters = 0.0
        avg_def_rules = 0.0
        n_arg_lines = 0.0
        avg_height_lines = 0.0
        avg_arg_lines = 0.0
        tree_numbers = 0
        ### Arguemnts, MDDL and Defeaters section ###
        dGraphs_data = result['dGraph']
        n_def_rules = 0  # To compute the average of defeasible rules
        for literal in dGraphs_data:
            # literal_key = literal.keys()[0]
            literal_key = list(literal.keys())[0]
            arguments = literal[literal_key]
            for argument in arguments:
                # argument_key = argument.keys()[0]
                argument_key = list(argument.keys())[0]
                if ',' in argument_key:
                    def_rules_in_body = len(argument[argument_key]["subarguments"])
                    if def_rules_in_body != 0: 
                        delete_presum = sum(1 if re.match('\[\([a-zA-Z]*\-\<[t|T]rue\)\]',subs) else 0 for subs in argument[argument_key]["subarguments"])
                        n_arguments += 1  
                        n_def_rules += def_rules_in_body - delete_presum
                        defeaters = argument[argument_key]['defeats']
                        n_defeaters += len(defeaters)
                    
        if n_arguments != 0:
            avg_def_rules = n_def_rules / n_arguments
        else:
            avg_def_rules = 0.0

        ### Trees section ###
        trees_data = result['status']
        for literal in trees_data:
            # literal_key = literal.keys()[0]
            literal_key = list(literal.keys())[0]
            trees = literal[literal_key]['trees']
            roots = [root for root in trees if len(root) == 2]
            lines = [attacks for attacks in trees if len(attacks) == 4] 
            if len(lines) != 0:
                for root in roots:
                    if '-<' in root[0]:
                        childs = [defeaters[3] for defeaters in lines if defeaters[2] == root[1]]
                        if len(childs) != 0:
                            #self.utils.print_msj("OK", str(lines))  
                            #self.utils.print_msj("OK", str(root))
                            n_arg_lines += self.count_lines(root[1], lines)
                            #self.utils.print_msj("OK", str(self.count_lines(root[1], lines)))
                            tree_numbers += 1
            else:
                pass
        sum_height_lines = sum(self.aux_height)
        self.aux_height = []
        #self.utils.print_msj("OK", "Suma altura líneas: " + str(sum_height_lines))
        #self.utils.print_msj("OK", "Número de líneas: " + str(n_arg_lines))
        #self.utils.print_msj("OK", "Trees: " + str(tree_numbers))
        if n_arg_lines != 0.0:
            avg_height_lines = sum_height_lines / n_arg_lines
        if tree_numbers != 0.0:
            avg_arg_lines = n_arg_lines / tree_numbers   # N° lines / N° Trees
        return {
            'n_arguments': int(n_arguments),
            'n_defeaters': int(n_defeaters),
            'avg_def_rules': float('{0:.2f}'.format(avg_def_rules)),
            'avg_arg_lines': float('{0:.2f}'.format(avg_arg_lines)),
            'avg_height_lines': float('{0:.2f}'.format(avg_height_lines))
        }


    def load(self, filePath):
        initial_time = time.time()
        core_response = self.query_delp_solver()
        end_time = time.time()
        query_time = end_time - initial_time
        self.times.append(query_time)
        if core_response != "Error":
            result = self.analyze_results(core_response)
            return result
        else:
            return {
                'n_arguments': 0,
                'n_defeaters': 0,
                'avg_def_rules': 0.0,
                'avg_arg_lines': 0,
                'avg_height_lines': 0.0
            }


    def compute_one(self) -> None:
        self.aux_height = []
        
        arguments = []
        defeaters = []
        def_rules = []
        arg_lines = []
        height_lines = []

        data = self.load(self.path_delp)
        arguments.append(data['n_arguments'])
        defeaters.append(data['n_defeaters'])
        def_rules.append(data['avg_def_rules'])
        arg_lines.append(data['avg_arg_lines'])
        height_lines.append(data['avg_height_lines'])
        
        results = {
            'arguments': arguments,
            'defeaters': defeaters,
            'def_rules': def_rules,
            'arg_lines': arg_lines,
            'height_lines': height_lines
        }
        
        self.utils.write_result(self.build_path_result(), results)

    #   'dataset_length' is the number of programs to analyze
    #   'result_file' is the name for the file with results
    def compute_dataset(self, dataset_length):
        global height_lines

        arguments = []
        defeaters = []
        def_rules = []
        arg_lines = []
        height_lines = []

        spinner = Spinner("Processing")
        for count in range(dataset_length):
            filePath = self.path_dataset + 'delp' + str(count) + '.delp'
            self.path_delp = filePath
            data = self.load(filePath)
            arguments.append(data['n_arguments'])
            defeaters.append(data['n_defeaters'])
            def_rules.append(data['avg_def_rules'])
            arg_lines.append(data['avg_arg_lines'])
            height_lines.append(data['avg_height_lines'])
            spinner.next()
        
        metric_args = sum(arguments) / len(arguments)
        mddl = sum(def_rules) / len(def_rules)
        t = sum(arg_lines) / len(arg_lines) 
        h = sum(height_lines) / len(height_lines)
        min_time = min(self.times)
        max_time = max(self.times)
        mean_time = sum(self.times) / len(self.times)

        results = {
            'arguments': metric_args,
            'mddl': mddl,
            't': t,
            'h': h,
            'times':{
                    'min': float('{:0.4f}'.format(min_time)),
                    'max': float('{:0.4f}'.format(max_time)),
                    'mean': float('{:0.4f}'.format(mean_time))
                } 
        }

        self.utils.write_result(self.build_path_result(), results)
