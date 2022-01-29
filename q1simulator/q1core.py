import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from functools import wraps


@dataclass
class Instruction:
    text_line_nr: int
    mnemonic: str
    args: Optional[Tuple[str]] = None
    label: Optional[str] = None

class Halt(Exception):
    pass

class Abort(Exception):
    pass

class Illegal(Exception):
    pass

class AsmSyntaxError(Exception):
    pass

def evaluate_args(arg_types, *, dest_arg=None):

    def decorator_args(func):
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):
            try:
                # print(f'{func.__name__} {args}')
                args = self._evaluate_args(arg_types, dest_arg, args)
            except Exception as ex:
                print(f'*** Exception: {ex}')
                print(f'*** Error in instruction {func.__name__[1:]} {",".join(args)}')
                print(f'*** Argument types: ({arg_types})')
                raise
            return func(self, *args)

        return func_wrapper

    return decorator_args



class Q1Core:
    def __init__(self, name, renderer):
        self.name = name
        self.renderer = renderer
        self.max_core_cycles= 10_000_000
        self.R = [0]*64
        self.iptr = 0
        self.instructions = []
        self.lines = []
        self.labels = {}
        self.errors = set()

    def load(self, program):
        labels = {}
        lines = program.split('\n')
        self.lines = lines

        instr = []
        for i,line in enumerate(lines):
            label, mnemonic, arglist = self.parseline(line)
            if label:
                icnt = len(instr)
                # print(f'label {label}:{icnt}')
                labels[label] = icnt
            if not mnemonic:
                continue
            instr.append(Instruction(i, mnemonic, arglist, label))

        self.labels = labels
        self.instructions = instr
        # pprint(instr)

    def parseline(self, line):
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
        re.fullmatch(log_pattern, command)
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

    def run(self):
        self.errors = set()
        self.R = [0]*64
        self.iptr = 0
        self.clock = CoreClock()

        if len(self.instructions) == 0:
            print(f'*** No instructions loaded')
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
            return

        try:
            cntr = 0
            while(True):
                cntr += 1
                instr = self.instructions[self.iptr]
                # print(f'({self.clock.core_time:4}) {self.lines[instr.text_line_nr]}')
                self.iptr += 1
                getattr(self, '_'+instr.mnemonic)(*instr.args)
                if self.iptr >= len(self.instructions):
                    raise Illegal(f'No instruction at {self.iptr:04}')
                if cntr >= self.max_core_cycles:
                    raise Abort('Core cycle limited exceeded',
                                'FORCED STOP')
        except Halt:
            rt_time_us = self.renderer.time / 1000
            logging.info(f'{self.name}: stopped ({cntr} cycles, {rt_time_us:7.3f} us)')
        except Illegal as ex:
            msg = f'Illegal instruction at line {self.iptr}: {ex}'
            self._print_error_msg(msg, instr, cntr)
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
        except Abort as ex:
            msg = f'Execution aborted: {ex.args[0]}'
            self._print_error_msg(msg, instr, cntr)
            self._error(ex.args[1])
        except AsmSyntaxError as ex:
            msg = f'Syntax error: {ex.args[0]}'
            self._print_error_msg(msg, instr, cntr)
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
        except:
            self._print_error_msg('Exception', instr, cntr)
            self._error('OOPS!!')
            raise


    def _print_error_msg(self, msg, instr, cntr):
        last_line = self.lines[instr.text_line_nr]
        rt_time_us = self.renderer.time / 1000
        print(f'*** {self.name}: {msg} ({cntr} cycles, {rt_time_us:7.3f} us)')
        print(f'*** Last instruction: {last_line}')

    def _error(self, msg):
        self.errors.add(msg)

    def _evaluate_args(self, arg_types, dest_arg, args):
        args = list(args)
        types = arg_types.split(',') if arg_types else []
        if len(args) != len(types):
            raise AsmSyntaxError(f'Incorrect number of arguments {len(args)}<>{len(types)}')
        select_imm = False
        allow_imm = True
        for i,arg in enumerate(args):
            allowed = types[i]
            c = arg[0]
            if i == dest_arg:
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
                # add 1 clock tick for every register
                self.clock.add_ticks(1)
                reg_nr = int(arg[1:])
                args[i] = self.R[reg_nr]
            else:
                if 'I' not in allowed:
                    raise AsmSyntaxError(f'Immediate operand not support as argument {i}')
                if 'R' in allowed:
                    select_imm = True
                args[i] = int(arg)

        if not allow_imm and select_imm:
            raise AsmSyntaxError('Combination of operands not supported')
        return args

    def _set_register(self, reg_nr, value):
        self.R[reg_nr] = value & 0xFFFF_FFFF
        # print(f'R{reg_nr} = {np.int32(np.uint32(self.R[reg_nr]))} ({self.R[reg_nr]:08X})')

    def print_registers(self, reg_nrs=None):
        if reg_nrs is None:
            reg_nrs = range(64)
        for reg_nr in reg_nrs:
            value = self.R[reg_nr]
            signed_value = ((value + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000
            float_value = signed_value / 2**31
            print(f'R{reg_nr:02}: {value:08X} {signed_value:11}  ({float_value:9.6f})')

    # === Below are Q1ASM opcode mnemonics with _ prefix.

    @evaluate_args('')
    def _illegal(self):
        raise Illegal('illegal instruction')

    @evaluate_args('')
    def _stop(self):
        raise Halt('stop instruction')

    @evaluate_args('')
    def _nop(self):
        self.clock.add_ticks(1)

    @evaluate_args('IRL')
    def _jmp(self, label):
        self.clock.add_ticks(4)
        self.iptr = label

    @evaluate_args('R,I,IRL')
    def _jlt(self, register, n, label):
        self.clock.add_ticks(4)
        if register < n:
            self.iptr = label

    @evaluate_args('R,I,IRL')
    def _jge(self, register, n, label):
        self.clock.add_ticks(4)
        if register >= n:
            self.iptr = label

    @evaluate_args('R,IRL', dest_arg=0)
    def _loop(self, register, label):
        self.clock.add_ticks(5)
        self._set_register(register, self.R[register] - 1)
        if self.R[register] != 0:
            self.iptr = label

    @evaluate_args('IRL,R', dest_arg=1)
    def _move(self, source, destination):
        self.clock.add_ticks(1)
        self._set_register(destination, source)

    @evaluate_args('IR,R', dest_arg=1)
    def _not(self, source, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, ~source)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _add(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs + rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _sub(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs - rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _and(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs & rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _or(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs | rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _xor(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs ^ rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _asl(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs << rhs)

    @evaluate_args('R,IR,R', dest_arg=2)
    def _asr(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs >> rhs)

    @evaluate_args('IR')
    def _set_mrk(self, value):
        self.clock.add_ticks(1)
        self.renderer.set_mrk(value)

    @evaluate_args('')
    def _reset_ph(self):
        self.clock.add_ticks(1)
        self.renderer.reset_ph()

    @evaluate_args('IR,IR,IR')
    def _set_ph(self, arg0, arg1, arg2):
        self.clock.add_ticks(1)
        self.renderer.set_ph(arg0, arg1, arg2)

    @evaluate_args('IR,IR,IR')
    def _set_ph_delta(self, arg0, arg1, arg2):
        self.clock.add_ticks(1)
        self.renderer.set_ph_delta(arg0, arg1, arg2)

    @evaluate_args('IR,IR')
    def _set_awg_gain(self, gain0, gain1):
        self.clock.add_ticks(1)
        self.renderer.set_awg_gain(gain0, gain1)

    @evaluate_args('IR,IR')
    def _set_awg_offs(self, offset0, offset1):
        self.clock.add_ticks(1)
        self.renderer.set_awg_offs(offset0, offset1)

    @evaluate_args('I')
    def _upd_param(self, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.upd_param(wait_after)

    @evaluate_args('IR,IR,I')
    def _play(self, wave0, wave1, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.play(wave0, wave1, wait_after)

    @evaluate_args('I,IR,I')
    def _acquire(self, bins, bin_index, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire(bins, bin_index, wait_after)

    @evaluate_args('I,IR,R,R,I')
    def _acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire_weighed(bins, bin_index, weight0, weight1, wait_after)

    @evaluate_args('IR')
    def _wait(self, time):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.wait(time)

    @evaluate_args('IR')
    def _wait_sync(self, wait_after):
        # assume wait_sync pauses the RT exec for at least 200 ns = 50 ticks
        self.clock.add_ticks(-50)
        self.renderer.wait_sync(wait_after)

    @evaluate_args('IR')
    def _wait_trigger(self, wait_after):
        raise NotImplementedError()

    @evaluate_args('IR')
    def _sw_req(self, value):
        raise NotImplementedError()

    # ---- Simulator commands ----

    def _log(self, msg, reg, options):
        if 'R' in options and reg.startswith('R'):
            reg_nr = int(reg[1:])
            value = self.R[reg_nr]
            signed_value = ((value + 0x8000_0000) & 0xFFFF_FFFF) - 0x8000_0000
            float_value = signed_value / 2**31
            if 'F' in options:
                value_str = f'{float_value:9.6f} ({value:08X})'
            else:
                value_str = f'{signed_value:11} ({value:08X})'
        else:
            value_str = ''

        time_str = ''
        if 'T' in options:
            time_str = f' q1:{self.clock.core_time:6} rt:{self.renderer.time:6} ns'
        print(f'{msg}: {value_str}{time_str}')


class CoreClock:
    def __init__(self):
        self.buffer = []
        self.core_time = 0

    def add_ticks(self, value):
        self.core_time += value * 4

    def schedule_rt(self, time):
        # print(f'Sched {time:6} at {self.core_time:6}')
        if time < self.core_time:
            print(f'*** Schedule {time:6} at {self.core_time:6} ***')
            raise Abort('Real time buffer underrun',
                        'SEQUENCE PROCESSOR RT EXEC ILLEGAL INSTRUCTION')

        b = self.buffer
        # remove executed entries.
        while len(b) and b[0] < self.core_time:
            b.pop(0)
        # q1core halts when buffer is full
        if len(b) >= 16:
            # q1core will continue when an instruction is read from buffer.
            # When q1core continues the time advantage is `time` - popped time.
            # So, core time will be equal to popped time
            # print(f'Stall {self.core_time} -> {b[0]}')
            self.core_time = b.pop(0)

        self.buffer.append(time)
