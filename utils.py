import json
import numpy as np
from typing import Union


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def my_round(number: np.ndarray) -> float:
    return float('{:0.2f}'.format(number))


def string_to_int_float(value: str):
    """
    Convert a string into a float or int
    Args:
        value: a number in string

    Returns:
        The float or int representation of a number
    """
    if value.isdigit():
        number = int(value)
    else:
        number = float(value)
    return number


def get_complement(literal: str) -> str:
    """
    Get the complement of a literal
    Args:
        -literal: The literal to obtain its complement
    """
    if '~' in literal:
        return literal.replace('~', '')
    else:
        return '~' + literal


def get_randint(a: int, b: int) -> int:
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


def get_choice(choices: Union[int, list]):
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


def get_random() -> float:
    """
    Generate and return a float number from [0,1)
    """
    return np.random.random()


def print_error(msg: str) -> None:
    print(Bcolors.FAIL + str(msg) + Bcolors.ENDC)


def print_info(msg: str) -> None:
    print(Bcolors.UNDERLINE + str(msg) + Bcolors.ENDC)


def print_ok(msg: str) -> None:
    print(Bcolors.OKGREEN + str(msg) + Bcolors.ENDC)


def to_string_decimal_format(number: float) -> str:
    """
    To convert a float number into string (changing the '.' with ',')
    Args:
        -number: The float number to be converted
    """
    return str(float('{0:.2f}'.format(number))).replace('.', ',')


def write_result(path_file: str, result: json) -> None:
    """
    To write the results in a json file
    Args:
        -path_file: The full path of the file to be created
        -result: The json data to be saved
    """
    with open(path_file, 'w') as output:
        json.dump(result, output, indent=4)


def get_data_from_file(path_file: str) -> json:
    """
    To read a json file
    Args:
        -path_file: The path of the file
    """
    try:
        file = open(path_file, "r")
        to_dict = json.load(file)
        file.close()
        return to_dict
    except IOError:
        # print(e)
        print_error("Error trying to open file: %s" % path_file)
        exit()
    except ValueError:
        # print(e)
        print_error("JSON incorrect format: %s" % path_file)
        exit()


def write_metrics(result_path: str, result_path_file: str):
    """
    To compute the average of all metrics and save in a
    simple txt file
    Args:
        -result_path: The path to save the metrics
        -result_path_files: The path of the file with all results
    """
    data = get_data_from_file(result_path_file)
    arguments = sum(data['arguments']) / len(data['arguments'])
    def_rules = sum(data['def_rules']) / len(data['def_rules'])
    arg_lines = sum(data['arg_lines']) / len(data['arg_lines'])
    height_lines = sum(data['height_lines']) / len(data['height_lines'])
    # Argument | ADDL | H | T
    to_write = ('\n'
                + to_string_decimal_format(arguments)
                + ' ' + to_string_decimal_format(def_rules)
                + ' ' + to_string_decimal_format(height_lines)
                + ' ' + to_string_decimal_format(arg_lines))

    with open(result_path + '/metrics.txt', 'a') as fileOutput:
        fileOutput.write(to_write)


def pretty(d, indent=0) -> None:
    """
    To print a dict object
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
