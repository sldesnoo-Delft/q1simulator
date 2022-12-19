import re

from dataclasses import dataclass
from typing import Optional, Tuple, Any, List

@dataclass
class Instruction:
    text_line_nr: int
    mnemonic: str
    args: Optional[Tuple[str]] = None
    label: Optional[str] = None
    reg_args: List[int] = None
    func_name: Optional[str] = None
    func: Any = None


class AsmSyntaxError(Exception):
    pass

# I = immediate, R = register, L = label, D = destination register
mnemonic_args = {
    'illegal': '',
    'stop': '',
    'nop': '',
    'jmp': 'IRL',
    'jlt': 'R,I,IRL',
    'jge': 'R,I,IRL',
    'loop': 'D,IRL',
    'move': 'IRL,D',
    'not': 'IR,D',
    'add': 'R,IR,D',
    'sub': 'R,IR,D',
    'and': 'R,IR,D',
    'or': 'R,IR,D',
    'xor': 'R,IR,D',
    'asl': 'R,IR,D',
    'asr': 'R,IR,D',
    'set_mrk': 'IR',
    'reset_ph': '',
    'set_freq': 'IR',
    'set_ph': 'IR',
    'set_ph_delta': 'IR',
    'set_awg_gain': 'IR,IR',
    'set_awg_offs': 'IR,IR',
    'upd_param': 'I',
    'play': 'IR,IR,I',
    'acquire': 'I,IR,I',
    'acquire_weighed': 'I,IR,IR,IR,I',
    'wait': 'IR',
    'wait_sync': 'IR',
    'wait_trigger': 'IR',
    'sw_req': 'IR',
    }


class Q1Parser:

    def __init__(self):
        self.labels = {}

    def parse(self, program):
        labels = {}
        lines = program.split('\n')
        self.lines = lines

        instructions = []
        for i,line in enumerate(lines):
            label, mnemonic, arglist = self._parseline(line)
            if label:
                icnt = len(instructions)
                labels[label] = icnt
            if not mnemonic:
                continue

            instructions.append(Instruction(i, mnemonic, arglist, label))

        self.labels = labels

        for instr in instructions:
            mnemonic = instr.mnemonic
            func_name = '_' + mnemonic
            instr.func_name = func_name
            if mnemonic in mnemonic_args:
                try:
                    args,reg_args = self._evaluate_args(mnemonic_args[mnemonic], instr.args)
                    instr.args = args
                    instr.reg_args = reg_args
                except AsmSyntaxError as ex:
                    print(ex)
                    print(lines[instr.text_line_nr])
                    raise

        return lines,instructions


    def _parseline(self, line):
        org_line = line
        label_pattern = r'(\w+:)'
        instr_pattern = r'(\w+:)?\s*(\w+)\s*(.*)'
        if line.startswith('#Q1Sim:'):
            return self._parse_simcmd(line[7:])
        try:
            end = line.index('#')
            line = line[:end]
        except:
            pass
        line = line.strip()
        if len(line) == 0:
            return [None, None, None]

        match = re.fullmatch(label_pattern, line)
        if match:
            label = match.group(1)
            label = label[:-1]
            return [label, None, None]

        match = re.fullmatch(instr_pattern, line)
        if match:
            label = match.group(1)
            if label:
                label = label[:-1]
            args = match.group(3).strip()
            if args:
                arglist = args.split(',')
            else:
                arglist = []
            return [label, match.group(2), arglist]
        raise Exception(f'{self.name}: Parse error on line: {org_line}')

    def _parse_simcmd(self, command):
        command = command.strip()
        # format: 'log "msg",register,options
        log_pattern = r'log "(.*)",(\w+)?,(\w+)?'
        match = re.fullmatch(log_pattern, command)
        if match:
            msg = match.group(1)
            register = match.group(2)
            options = match.group(3)
            if msg is None:
                msg = ''
            return None,'log',(msg,register,options)
        print(f'Unknown simulator command:{command}')
        return None,None,None

    def _evaluate_args(self, arg_types, args):
        types = arg_types.split(',') if arg_types else []
        args = list(args)
        if len(args) != len(types):
            raise AsmSyntaxError(f'Incorrect number of arguments {len(args)}<>{len(types)}')
        select_imm = False
        allow_imm = True
        reg_args = []
        for i,arg in enumerate(args):
            allowed = types[i]
            c = arg[0]
            if allowed == 'D':
                if c != 'R':
                    raise AsmSyntaxError('Destination must be register')
                reg_nr = int(arg[1:])
                args[i] = reg_nr
            elif c == '@':
                if 'L' not in allowed:
                    raise AsmSyntaxError(f'Label operand not support as argument {i}')
                if 'R' in allowed:
                    select_imm = True
                try:
                    line_nr = self.labels[arg[1:]]
                    args[i] = line_nr
                except:
                    raise AsmSyntaxError(f'Label {arg} not defined')
            elif c == 'R':
                if 'R' not in allowed:
                    raise AsmSyntaxError(f'Register operand not support as argument {i}')
                if 'I' in allowed:
                    allow_imm = False
                reg_nr = int(arg[1:])
                args[i] = reg_nr
                reg_args.append(i)
            else:
                if 'I' not in allowed:
                    raise AsmSyntaxError(f'Immediate operand not support as argument {i}')
                if 'R' in allowed:
                    select_imm = True
                args[i] = int(arg)

        if not allow_imm and select_imm:
            raise AsmSyntaxError('Combination of operands not supported')

        if len(reg_args) == 0:
            reg_args = None
        return args,reg_args
