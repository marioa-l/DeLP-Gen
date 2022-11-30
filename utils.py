import json
import matplotlib.pyplot as pyplot
import numpy as np
from typing import Union

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Utils: 

    def __init__(self):
        pass
    

    def print_error(self, msg: str) -> None:
        print(bcolors.FAIL + str(msg) + bcolors.ENDC)


    def print_info(self, msg: str) -> None:
        print(bcolors.UNDERLINE + str(msg) + bcolors.ENDC)


    def print_ok(self, msg: str) -> None:
        print(bcolors.OKGREEN + str(msg) + bcolors.ENDC)

   
    def to_string_decimal_format(self, number: float) -> str:
        """
        To convert a float number into string (changing the '.' with ',')
        Args:
            -number: The float number to be converted
        """
        return str(float('{0:.2f}'.format(number))).replace('.',',')


    def write_result(self, path_file: str, result: json) -> None:
        """
        To write the results in a json file
        Args:
            -path_file: The full path of the file to be created
            -result: The json data to be saved
        """
        with open(path_file, 'w') as output:
            json.dump(result, output, indent=4)

    
    def get_data_from_file(self, path_file: str) -> json:
        """
        To read a json file
        Args:
            -path_file: The path of the file
        """
        try:
            file = open(path_file, "r")
            toDict = json.load(file)
            file.close()
            return toDict
        except IOError:
            # print(e)
            self.print_msj("ERROR", "Error trying to open file: %s" 
                            % (path_file))
            exit()
        except ValueError:
            # print(e)
            self.print_msj("ERROR", "JSON incorrect format: %s" 
                            % (path_file))
            exit()


    def write_metrics(self, result_path: str, result_path_file: str):
        """
        To comput the average of all metrics and save in a
        simple txt file (to use in excel)
        Args:
            -result_path: The path to save the metrics
            -result_path_files: The path of the file with all results
        """
        data = self.get_data_from_file(result_path_file)
        arguments = sum(data['arguments']) / len(data['arguments'])
        def_rules = sum(data['def_rules']) / len(data['def_rules'])
        arg_lines = sum(data['arg_lines']) / len(data['arg_lines']) 
        height_lines = sum(data['height_lines']) / len(data['height_lines']) 
        # Argument | MDDL | H | T    
        to_write = ('\n'
                    + self.to_string_decimal_format(arguments)
                    + ' ' + self.to_string_decimal_format(def_rules)
                    + ' ' + self.to_string_decimal_format(height_lines)
                    + ' ' + self.to_string_decimal_format(arg_lines))

        with open(result_path + '/metrics.txt', 'a') as fileOutput:
            fileOutput.write(to_write)


    def get_random(self) -> float:
        """
        Generate and return a float number from [0,1)
        """
        return np.random.random()


    def get_choice(self, choices: Union[int, list]):
        """
        Get and return an element from a list or from [0,choices)
        (if choices is an int)
        Args:
            -choices: List of elements or an int
        """
        if isinstance(choices, list):
            if len(choices) == 1:
                return choices[0]
            else:
                return np.random.choice(choices, 1)[0]
        return np.random.choice(choices, 1)[0]


    def get_randint(self, a: int, b: int) -> int:
        """
        Get a random int from [a,b]
        Args:
            a: lower bound
            b: upper bound
        """
        if a == b:
            return a
        else:
            return np.random.randint(a, b + 1, 1)[0]


    def get_complement(self, literal: str) -> str:
        """
        Get the complement of a literal
        Args:
            -literal: The literal to obtain its complement
        """
        if '~' in literal:
            return literal.replace('~', '')
        else:
            return '~' + literal


    def pretty(self, d, indent=0) -> None:
        """
        To print a dect object
        """
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                self.pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))


    def string_to_int_float(self, value: str):
        number = '0'
        if value.isdigit():
            number = int(value)
        else:
            number = float(value)
        return number

    def my_round(self, number: float) -> float:
        return float('{:0.2f}'.format(number))
