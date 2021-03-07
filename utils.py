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
        if msg_type == 'ERROR':
            print(bcolors.FAIL + text + bcolors.ENDC)
        elif msg_type == 'INFO':
            print(bcolors.UNDERLINE + text + bcolors.ENDC)
        elif msg_type == 'OK':
            print(bcolors.OKGREEN + text + bcolors.ENDC)

    
    def write_result(self, path_file: str, result: json) -> None:
        with open(path_file, 'w') as output:
            json.dump(result, output, indent=4)

    
    def get_data_from_file(self, path_file):
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
