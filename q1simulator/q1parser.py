import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Instruction:
    text_line_nr: int
    mnemonic: str
    arglist: tuple[str] | None = None
    label: str | None = None
    args: list[int | str] | None = None
    reg_args: list[int] | None = None
    func_name: str | None = None
    func: any = None
    clock_ticks: int = 1


class AsmSyntaxError(Exception):
    pass


# S = immediate signed, U = immediate unsigned, R = register, L = label, D = destination register
mnemonic_args = {
    'illegal': '',
    'stop': '',
    'nop': '',
    'jmp': 'URL',
    'jlt': 'R,U,URL',
    'jge': 'R,U,URL',
    'loop': 'D,URL',
    'move': 'URL,D',
    'not': 'UR,D',
    'add': 'R,UR,D',
    'sub': 'R,UR,D',
    'and': 'R,UR,D',
    'or': 'R,UR,D',
    'xor': 'R,UR,D',
    'asl': 'R,UR,D',
    'asr': 'R,UR,D',
    'set_mrk': 'UR',
    'reset_ph': '',
    'set_freq': 'SR',
    'set_ph': 'UR',
    'set_ph_delta': 'UR',
    'set_awg_gain': 'SR,SR',
    'set_awg_offs': 'SR,SR',
    'set_cond': 'UR,UR,UR,U',
    'upd_param': 'U',
    'play': 'UR,UR,U',
    'acquire': 'U,UR,U',
    'acquire_weighed': 'U,UR,UR,UR,U',
    'set_latch_en': 'UR,U',
    'latch_rst': 'UR',
    'wait': 'UR',
    'wait_sync': 'UR',
    'wait_trigger': 'UR',
    }


class Q1Parser:

    def __init__(self):
        self.labels = {}

    def parse(self, program):
        labels = {}
        lines = program.split('\n')
        self.lines = lines

        instructions = []
        for i, line in enumerate(lines):
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
                    args, reg_args = self._evaluate_args(mnemonic_args[mnemonic], instr.arglist)
                    instr.args = args
                    instr.reg_args = reg_args
                    if reg_args and mnemonic not in ['jmp', 'jge', 'jlt', 'loop']:
                        # 1 cycle for every register. instr and 1st register are loaded in 1 cycle
                        instr.clock_ticks = len(reg_args)
                    else:
                        # 1 cycle to load instruction
                        instr.clock_ticks = 1
                except AsmSyntaxError as ex:
                    print(ex)
                    print(lines[instr.text_line_nr])
                    raise
                except Exception as ex:
                    print(ex)
                    print(lines[instr.text_line_nr])
                    raise
            else:
                instr.args = instr.arglist

        return lines, instructions

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
                arglist = args.replace(' ','').split(',')
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
        trigger_pattern = r'sim_trigger (\d),\s*([01])'
        match = re.fullmatch(trigger_pattern, command)
        if match:
            addr = match.group(1)
            value = match.group(2)
            return None,'sim_trigger',(addr, value)
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
        for i, arg in enumerate(args):
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
                if 'U' in allowed or 'S' in allowed:
                    allow_imm = False
                reg_nr = int(arg[1:])
                args[i] = reg_nr
                reg_args.append(i)
            else:
                if 'R' in allowed:
                    select_imm = True
                if 'U' in allowed:
                    args[i] = np.uint32(arg)
                elif 'S' in allowed:
                    args[i] = np.int32(arg)
                else:
                    raise AsmSyntaxError(f'Immediate operand not support as argument {i}')

        if not allow_imm and select_imm:
            raise AsmSyntaxError('Combination of operands not supported')

        if len(reg_args) == 0:
            reg_args = None
        return args, reg_args
