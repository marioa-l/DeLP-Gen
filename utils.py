import json
import matplotlib.pyplot as pyplot

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
        
    
    def print_msj(self, msg_type: str, text: str) -> None:
        """
        To print messages with differents colors
        Args:
            -msg_type: 
                'Error': To show an error
                'Info': To show information
                'OK': To show correct outputs
            -text: The text to be printed
        """
        if msg_type == 'ERROR':
            print(bcolors.FAIL + text + bcolors.ENDC)
        elif msg_type == 'INFO':
            print(bcolors.UNDERLINE + text + bcolors.ENDC)
        elif msg_type == 'OK':
            print(bcolors.OKGREEN + text + bcolors.ENDC)

   
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


    def write_metrics(self, result_path_file: str):
        """
        To comput the average of all metrics and save in a
        simple txt file (to use in excel)
        Args:
            result_path_files: The path of the file with all results
        """
        data = self.get_data_from_file(result_path_file)
        arguments = sum(data['arguments']) / len(data['arguments'])
        def_rules = sum(data['def_rules']) / len(data['def_rules'])
        arg_lines = sum(data['arg_lines']) / len(data['arg_lines']) 
        height_lines = sum(data['height_lines']) / len(data['height_lines'])

        to_write = ('Arguments MDDL H T\n'
                    + self.to_string_decimal_format(arguments)
                    + ' ' + self.to_string_decimal_format(def_rules)
                    + ' ' + self.to_string_decimal_format(height_lines)
                    + ' ' + self.to_string_decimal_format(arg_lines))

        with open('./metrics.txt', 'w') as fileOutput:
            fileOutput.write(to_write)
