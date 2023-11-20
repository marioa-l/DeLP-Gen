import glob
import copy
from utils import *

"""
Hyperparams

# Minimum number of base elements
BE
    
# Percentage of facts in the base
FACTS

# Percentage of negated literals ()not yet implemented)
NEG_PROB

# Percentage of defeasible rules
DRUL

#Maximum number of rules with the same head literal
(HEADS <= ARGLVL)
HEADS

#Maximum number of literals in the rule’s body
BODY

#Minimum number of distinct arguments for each level
ARGLVL

#Maximum argument level that can be reached
LVL

#Maximum number of defeaters for an argument
DEFT

#Height of dialectical trees
HEIGHT

# Probability of attack a inner point of an argument (not yet implemented)
INNER_PROB

# Number of programs to generate
N_PROGRAMS
"""


def find_rule(conclusion: str, complete_argument: list) -> list:
    """
    Find and return the body literals of a rule in a complete argument
    with conclusion <conclusion> and its type (drule o srule)
    Args:
        -conclusion: The conclusions of argument
        -complete_argument: The complete argument
    Output:
        -[body_literals, tipo]:
            --body_literals: The literals of the body
            --tipo: 'drule' or 'srule'
    """
    try:
        rule = next(rule for rule in complete_argument if rule[0] == conclusion)
        literals_tipo = [set(rule[2]), rule[1]]
        return literals_tipo
    except StopIteration:
        # Facts or Presumptions
        return [set(), '']


def build_actntactsets(conclusion: str, complete_argument: list) -> dict:
    """
    Compute the Act-sets and NTAct-sets of the argument <complete_argument>
    Args:
        -complete_argument: The complete argument
        -conclusion: The conclusion of the argument
    Output:
        -dict:{
            'act_sets': set,
            'ntact_sets': set
        }
    """
    c = [[{conclusion}, 'trivial']]
    act_sets = []
    ntact_sets = []
    exp_sets = []
    while True:
        [conj, tipo] = c.pop(0)
        for lit in conj:
            [new_actset, type_rule] = find_rule(lit, complete_argument)
            if len(new_actset) != 0:
                if tipo == 'trivial' and type_rule == 'srule':
                    # The new act set is 'trivial'
                    type_new_actset = 'trivial'
                else:
                    # The new act set is 'no trivial'
                    type_new_actset = 'no trivial'
                if new_actset not in exp_sets:
                    c.append([new_actset, type_new_actset])
        act_sets.append(conj)
        if tipo == 'no trivial':
            ntact_sets.append(conj)
        exp_sets.append(conj)
        if len(c) == 0:
            break
    return {
            'act_sets': act_sets,
            'ntact_sets': ntact_sets
            }


def is_defeasible(complete_argument: list) -> bool:
    """
    Determine if an argument is defeasible or not
    Args:
        -complete_argument: The complete argument
    """
    try:
        next(rule for rule in complete_argument if rule[1] == 'drules')
        return True
    except StopIteration:
        return False


class Generator:

    # Symbols
    srule_symbol = '<-'
    drule_symbol = '-<'

    def __init__(self) -> None:
        """
        Constructor
        """
        
        """
        Define default values for all hyper parameters
        """ 
        self.params = {
                "BE": 5,
                "FACTS": 0.5,
                "NEG_PROB": 0.5,
                "DRUL": 0.5,
                "HEADS": 1,
                "BODY": 1,
                "ARGLVL": 1,
                "LVL": 1,
                "DEFT": 1,
                "HEIGHT": 1,
                "INNER_PROB": 0.5,
                "N_PROGRAMS": 1,
                "PREF_CRITERION": "more_specific"
                }

        """
        Global values and assistant structures
        """
        # Index of literals
        self.COUNT_LIT = 0
        # Used heads 
        self.USED_HEADS = ()
        # Used for control of defeaters building
        self.USED_NTACTSETS = []
        # To save all used literals
        self.LITERALS = {}
        """"""
        
        """
        Structure to save all rules
        """
        self.rules = {
                'drules': [],
                'srules': [],
                'facts': [],
                'presumptions': []
                } 

        # The KB (program)
        self.levels = {}
        """"""

    def define_hyperparams(self, hyperparams) -> None:
        self.params = copy.copy(hyperparams)

    def clear_datastructures(self) -> None:
        """
        Clear all data structures and global values
        """
        self.COUNT_LIT = 0
        self.USED_HEADS = ()
        self.USED_NTACTSETS = []
        self.LITERALS = {}
        self.rules = {
                'drules': [],
                'srules': [],
                'facts': [],
                'presumptions': []
                }
        self.levels = {}

    def get_new_id(self) -> str:
        """
        Return an id for literal and update the counter
        """
        id_literal = str(self.COUNT_LIT)
        self.COUNT_LIT += 1
        return id_literal

    def create_strict_rule(self, head: str, body: Union[list, tuple]) -> str:
        """
        Create a strict rule.
        Args:
            -head: A literal (the head of the stric rule)
            -body: A list of literals (the body of the strict rule)
        """ 
        if isinstance(body[0], str):
            # Is fact
            body_string = 'true'
        else:
            # Is rule
            body_literals = self.get_body_literals(body)
            body_string = ','.join(body_literals)

        return str(head + ' ' + self.srule_symbol + ' ' + body_string + '.')

    def create_def_rule(self, head: str, body: Union[list, tuple]) -> str:
        """
        Create a defeasible rule.
        Args:
            -head: A literal (the head of the defeasible rule)
            -body: A list of literals (the body of the defeasible rule)
        """
        if isinstance(body[0], str):
            if body[0] != 'true':
                body_string = ','.join(body)
            else:
                # Is fact or presumption
                body_string = 'true'
        else:
            # Is rule
            body_literals = self.get_body_literals(body)
            body_string = ','.join(body_literals)

        return str(head + ' ' + self.drule_symbol + ' ' + body_string + '.')

    def get_head(self, level: int, tipo: str, pos: int) -> str:
        """
        Return the head of a rule in the KB
        Args:
            -level: The level of the rule
            -tipo: The type of rule (drules o srules)
            -pos: The index of the rule inside the type and level
        """
        return self.levels[level][tipo][pos][0]

    def get_body_literals(self, body) -> list:
        """
        Get all literals in a body
        Args:
            -body: A list of tuples, each tuple is a rule in the body argument
        Output:
            - A list with all literals in body 
        """
        body_literals = []

        for tup_info in body:
            literal = self.levels[tup_info[1]][tup_info[0]][tup_info[2]][0]
            body_literals.append(literal)
        return body_literals

    def get_new_literal(self) -> str:
        """
        Create and return a new literal for the program
        Args:--
        Output:
            - A string that represent a literal 
        """
        polarity = get_random()
        atom = 'a_' + self.get_new_id()
        if polarity < self.params["NEG_PROB"]:
            literal = get_complement(atom)
        else:
            literal = atom
        return literal

    def add_to_kb(self, level: int, head: str, body: list, r_type: str) -> None:
        """
        Add a new argument to the KB
        Args:
            -level: The level to which the argument belongs
            -head: The head of the argument
            -body: Body of the argument
            -r_type: The type of rule with <head> as its consequent. Options:
                --rnd: Randomly assign whether it will be a strict rule or
                a defeasible rule (considering the params DEF_PROB).
                --srules: Save as strict rule
                --drules: Save as defeasible rule
        """
        if r_type == 'rnd':
            random_ds = get_random()
            if random_ds < self.params["DRUL"]:
                self.levels[level]['drules'].append((head, body))
                self.rules['drules'].append(head)
                self.LITERALS[level].append(head)
            else:
                self.levels[level]['srules'].append((head, body))
                self.rules['srules'].append(head)
                self.LITERALS[level].append(head)
        else:
            self.levels[level][r_type].append((head, body))
            self.rules[r_type].append(head)
            self.LITERALS[level].append(head)

    def get_one_rule_level(self, level: int) -> tuple:
        """
        Select a rule from possibles rules to build an argument body
        Args:
            -level: A KB level
        Out:
            -tuple: A tuple of the form (type, level, pos)
                --type: 'drule' or 'srule'
                --level: Level of the rule
                --pos: Index of the rule in the level <level>
        """
        possibles_drules = [index for index, drule in enumerate(self.levels[level]["drules"]) if drule[0] not in
                            self.USED_HEADS]
        possibles_srules = [index for index, srule in enumerate(self.levels[level]["srules"]) if srule[0] not in
                            self.USED_HEADS]
        random_ds = get_random()
        if random_ds < self.params["DRUL"]:
            if len(possibles_drules) != 0:
                # Take a drule (its position) from level <level>
                index_drule = get_choice(possibles_drules)
                rule = ('drules', level, index_drule)
            else:
                # Build a drule and put into the level <level>?
                # No more drules
                lit = self.get_new_literal()
                self.levels[0]['drules'].append((lit, ('true',)))
                self.LITERALS[0].append(lit)
                rule = ('drules', 0, len(self.levels[0]['drules']) - 1)
        else:
            if len(possibles_srules) != 0:
                # Take a srule (its position) from level <level>
                index_srule = get_choice(possibles_srules)
                rule = ('srules', level, index_srule)
            else:
                # Build a srule and put into the level <level>?
                # No more srules!
                lit = self.get_new_literal()
                self.levels[0]['srules'].append((lit, ('true',)))
                self.LITERALS[0].append(lit)
                rule = ('srules', 0, len(self.levels[0]['srules']) - 1)
        return rule

    def build_body(self, level: int, conclusion: str):
        """
        Build an argument body (a list of tuples) whit at least one rule from
        a particular KB level.
        Args:
            -level: A KB level
            -conclusion: The conclusion of the argument for which we are 
            creating a body
        """
        body = ()
        # To not add the conclusion or its complement in the body 
        self.USED_HEADS = self.USED_HEADS + (conclusion,)
        complement_conclusion = get_complement(conclusion)
        self.USED_HEADS = self.USED_HEADS + (complement_conclusion,)

        body_size = get_randint(1, self.params["BODY"])
        rule = self.get_one_rule_level(level - 1)
        rule_head = self.get_head(rule[1], rule[0], rule[2])
        self.USED_HEADS = self.USED_HEADS + (rule_head,)
        body = body + (rule,)

        for aux in range(body_size - 1):
            select_level = get_choice(level)
            new_rule = self.get_one_rule_level(select_level)
            new_rule_head = self.get_head(new_rule[1], new_rule[0], new_rule[2])
            self.USED_HEADS = self.USED_HEADS + (new_rule_head,)
            body = body + (new_rule,) 
        
        self.USED_HEADS = ()
        return body 

    def clean_level(self, level: int) -> None:
        """
        Delete all duplicate in level <level>
        Args:
            -level: The level to clean
        """
        self.levels[level]['drules'] = list(set(self.levels[level]['drules']))
        self.levels[level]['srules'] = list(set(self.levels[level]['srules']))

    def build_arguments(self, level: int) -> None:
        """"
        Build all arguments for a particular level
        Args:
            -level: A KB level
        """
        # To save all arguments in this level
        self.levels[level] = {'drules': [], 'srules': []}
        # To save all literals
        self.LITERALS[level] = []
        # Min number of different arguments in the level
        min_args = self.params["ARGLVL"]

        for i_aux in range(min_args):
            # Generate a new head (conclusion)
            head = self.get_new_literal()
            # To define how many arguments with the same head to create
            args_head = get_randint(1, self.params["HEADS"])
            # Build all arguments for <head>
            for j_aux in range(args_head):
                # Generate the body of the argument
                body = self.build_body(level, head)
                # Add to the KB
                self.add_to_kb(level, head, body, 'rnd') 
        self.clean_level(level)
   
    def build_complete_arguments(self, argument: list, tipo: str) -> list:
        """
        Build the complete argument
        Args:
            -argument: The argument as list of tuples (head, (body))
            -tipo:
                --'srule': The rule that derive the conclusion is strict
                --'drule': The rule that derive the conclusion is defeasible
        """
        if not isinstance(argument[1][0], str):
            body_literals = self.get_body_literals(argument[1])
            rule = [(argument[0], tipo, body_literals)]

            for in_body in argument[1]:
                arg = self.levels[in_body[1]][in_body[0]][in_body[2]]
                rule += self.build_complete_arguments(arg, in_body[0])
            return rule
        else:
            # Is fact or presumption
            return []

    def build_defeater(self, head: str, ntact_sets: list, type: str) -> list:
        if len(ntact_sets) != 0 and ntact_sets[0] != '':
            new_def_lit = self.get_new_literal().replace('a', 'd')
            self.add_to_kb(0, new_def_lit, ('true',), 'drules')
            ntact_def = copy.copy(get_choice(ntact_sets))
            ntact_def.add(new_def_lit)
            body_def = list(ntact_def)
            head_def = get_complement(head)
            self.add_to_kb(self.params["LVL"] + 1, head_def, body_def, 'drules')
            self.USED_NTACTSETS.append(ntact_def)
            return [head_def, [ntact_def]]
        else:
            # Argument without ntactsets (the base of the KB)
            return []

    def build_dialectical_trees(self) -> None:
        """
        To build all dialectical trees for arguments in the top level of the
        KB
        """
        self.levels[self.params["LVL"] + 1] = {'drules': [], 'srules': []}
        self.LITERALS[self.params["LVL"] + 1] = []
        tree_height = self.params["HEIGHT"]
        ramification = self.params["DEFT"]
        for tipo, args in self.levels[self.params["LVL"]].items():
            for argument in args: 
                complete_arg = self.build_complete_arguments(argument, tipo)
                if is_defeasible(complete_arg):
                    actntact_sets = build_actntactsets(argument[0], complete_arg)
                    self.build_tree(argument[0], actntact_sets["ntact_sets"], tree_height, ramification)

    def build_tree(self, head: str, ntact_sets: list, height: int, ram: int) -> None:
        """
        Build the dialectical tree for a particular argument
        Args:
            -head: The conclusion of the root argument
            -ntact_sets: A list with all the ntact sets of the root argument
            -height: The height of the tree to be constructed
            -ram: Ramification factor (number of different defeats for an 
            argument)
        """
        if height == 1:
            # Build the leaves of the tree
            for aux in range(ram):
                self.build_defeater(head, ntact_sets, 'blocking')
        else:
            # Internal levels of the dialectical tree
            defeaters = []
            for aux in range(ram):
                defeater = self.build_defeater(head, ntact_sets, 'blocking') 
                defeaters.append(defeater)
            self.USED_NTACTSETS = []       
            for defeater in defeaters:
                if len(defeater) != 0: 
                    ram = get_randint(1, ram)
                    self.build_tree(defeater[0], defeater[1], height - 1, ram)

    def build_kb_base(self) -> None:
        """
        Build the KB Base (facts and presumptions only)
        """
        self.levels[0] = {'drules': [], 'srules': []}
        self.LITERALS[0] = []
        for i in range(self.params["BE"]):
            random_fp = get_random()
            literal = self.get_new_literal()
            if random_fp < self.params["FACTS"]:
                # New Fact
                self.levels[0]['srules'].append((literal, ('true',)))
                self.LITERALS[0].append(literal)
            else:
                # New Presumption
                self.levels[0]['drules'].append((literal, ('true',)))
                self.LITERALS[0].append(literal)

    def build_kb(self, level: int) -> None:
        """
        Build the KB (program) from level 1 to <level>
        Args:
            -level: Total number of KB levels
        """
        if level == 1:
            self.build_arguments(level)
        else:
            self.build_kb(level - 1)
            self.build_arguments(level)

    def to_delp_format(self) -> str:
        """
        Return the delp program in string and json format
        """
        program = []
        delp_json = []
        to_string = 'use_criterion(' + self.params["PREF_CRITERION"] + ').'
        program.append('')
        for key, value in self.levels.items():
            kb_level = str(key)
            program.append('\n/*** KB LEVEL = ' + kb_level + ' ***/')
            for drule in value['drules']:
                rule = self.create_def_rule(drule[0], drule[1])
                if rule not in program:
                    program.append(rule)
                    delp_json.append(rule)
            for srule in value['srules']:
                rule = self.create_strict_rule(srule[0], srule[1])
                if rule not in program:
                    program.append(rule)
                    delp_json.append(rule)

        for rule in program:
            to_string += rule + '\n'

        return [to_string, delp_json]


    def write_delp_program(self, result_path: str, id_program: int) -> None:
        """
        Save or return the delp program
        Args:
            result_path: The path for save the program
            id_program: The id of the program
        """ 
        [to_string, delp_json] = self.to_delp_format()
        with open(result_path + str(id_program) + 'delp' + '.delp', 'w') as outfile:
            outfile.write(to_string)
        
        filtered_literals =  {k: v for k, v in self.LITERALS.items() if v != []}
        write_result(result_path + str(id_program) + 'delp' + '.json', {
                    'delp': delp_json,
                    'literals': filtered_literals
                    })

    def generate(self, result_path: str, t_output: str, hyperparams =
            'undefined') -> list:
        """
        Generate a delp program with hyperparams (if they are in <args>)
        Args:
            -result_path: The path to save the program
            -hyperparams: Hyperparams
        """
        if hyperparams != 'undefined':
            self.define_hyperparams(hyperparams)
        n_files = len(glob.glob(result_path + '*.delp'))
        if t_output == 'write':
            for id_program in range(n_files, self.params["N_PROGRAMS"] + n_files):
                self.clear_datastructures()
                self.build_kb_base()
                self.build_kb(self.params["LVL"])
                self.build_dialectical_trees()
                self.write_delp_program(result_path, id_program)
            return []
        else:
            generated_programs = []
            for id_program in range(n_files, self.params["N_PROGRAMS"] + n_files):
                self.clear_datastructures()
                self.build_kb_base()
                self.build_kb(self.params["LVL"])
                self.build_dialectical_trees()
                program = self.to_delp_format()[0]
                generated_programs.append(program)
            return generated_programs

