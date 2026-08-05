import re
from dataclasses import dataclass
from typing import Any

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
    func: Any = None
    clock_ticks: int = 1


class AsmSyntaxError(Exception):
    def __str__(self):
        # lines = [f"{self.__class__.__name__}:"] + list(self.args)
        return "\n\t" + "\n\t".join(self.args)


# ? = optional argument
# b = 1 bit (boolean)
# A = 14 bit address
# I = immediate signed or unsigned 32 bit
# S = immediate signed 32 bit
# s = immediate signed 16 bit
# U = immediate unsigned 32 bit
# u = immediate unsigned 16 bit
# R = register (note 1)
# r = register (note 2)
# L = label
# D = destination register
# note 1: if argument can be 'R' or immediate, then this selection must apply to all arguments.
# note 2: if argument can be 'r' or immediate, then at least one of the arguments must be a register (arithmetic).


# Note: first character is integer type.
mnemonic_args_v1 = {
    "illegal": "",
    "stop": "",
    "nop": "",
    "jmp": "URL",
    "jlt": "R,U,URL",
    "jge": "R,U,URL",
    "loop": "D,URL",
    "move": "URL,D",
    "not": "UR,D",
    "add": "R,UR,D",
    "sub": "R,UR,D",
    "and": "R,UR,D",
    "or": "R,UR,D",
    "xor": "R,UR,D",
    "asl": "R,UR,D",
    "asr": "R,UR,D",
    "set_mrk": "uR",
    "reset_ph": "",
    "set_freq": "SR",
    "set_ph": "UR",
    "set_ph_delta": "UR",
    "set_awg_gain": "sR,sR",
    "set_awg_offs": "sR,sR",
    "set_cond": "uR,uR,uR,u",
    "upd_param": "u",
    "play": "uR,uR,u",
    "acquire": "u,uR,u",
    "acquire_weighed": "u,uR,uR,uR,u",
    "acquire_ttl": "u,uR,u,u",
    "set_latch_en": "uR,u",
    "latch_rst": "uR",
    "wait": "uR",
    "wait_sync": "uR",
    "wait_trigger": "uR",
}


# Note: first character is integer type.
mnemonic_args_v2 = {
    "illegal": "",
    "stop": "S?",
    "nop": "",
    "jmp": "ARL",
    "jz": "ARL",
    "jnz": "ARL",
    "jo": "ARL",
    "jno": "ARL",
    "js": "ARL",
    "jns": "ARL",
    "jg": "ARL",
    "jge": "ARL",
    "jl": "ARL",
    "jle": "ARL",
    "ja": "ARL",
    "jae": "ARL",
    "jb": "ARL",
    "jbe": "ARL",

    "move": "IRL,D",
    "not": "IR,D",
    "add": "Ir,Ir,D",
    "sub": "Ir,Ir,D",
    "cmp": "Ir,Ir",

    "mulu16": "ur,ur,D",
    "muls16": "sr,sr,D",
    "mulu32": "Ur,Ur,D,D",
    "mulu32l": "Ur,Ur,D",
    "mulu32h": "Ur,Ur,D",
    "muls32": "Sr,Sr,D,D",
    "muls32l": "Sr,Sr,D",
    "muls32h": "Sr,Sr,D",

    "and": "Ir,Ir,D",
    "or": "Ir,Ir,D",
    "xor": "Ir,Ir,D",
    "asl": "Ir,Ir,D",
    "asr": "Ir,Ir,D",
    "lsl": "Ir,Ir,D",
    "lsr": "Ir,Ir,D",
    "test": "Ir,Ir",

    "set_mrk": "uR",
    "reset_ph": "",
    "set_freq": "SR",
    "set_ph": "UR",
    "set_ph_delta": "UR",
    "set_awg_gain": "sR,sR",
    "set_awg_offs": "sR,sR",
    # "set_time_ref": "",
    "set_scope_en": "bR",
    "set_cond": "bR,uR,uR,u",

    "upd_param": "uR",
    "play": "uR,uR,u",
    "acquire": "u,UR,u",
    "acquire_weighted": "u,UR,uR,uR,u",
    "acquire_ttl": "u,UR,b,u",
    "set_latch_en": "bR,u",
    "latch_rst": "uR",
    "wait": "uR",
    "wait_sync": "uR",
    "wait_trigger": "uR,uR",

    # "acquire_timetags": "u,UR,u,uR,u",
    # "acquire_digital": "u,UR,u",
    # "upd_thres": "u,U,u"

    # FEEDBACK Instructions:
    # "fb_acq_tb_id": "u,u",
    # "fb_acq_tb_cfg": "u,u,u,u,u",
    # "fb_acq_iq_id": "u,u",
    # "fb_acq_tb_valid": "n,u",
    # "fb_acq_tb_extra": "b,u,u",
    # "fb_acq_tb_mock_en": "b,b,b,u",
    # "fb_acq_iq_shift": "u,u",
    # "fb_com_data": "u,U,u",
    # "fb_com_cfg": "u,u,u,u",
    # "fb_com_extra": "u,u,u",
    # "fb_pop_data": "u,R",
    # "fb_pull_data": "R,R",

    # DEPRECATED:
    "depr_jlt": "R,U,URL",
    "depr_jge": "R,U,URL",
    "depr_loop": "D,URL",
    "depr_acquire_weighed": "u,UR,uR,uR,u",
}


class Q1Parser:

    def __init__(self, v2: bool = False):
        self.labels = {}
        self.v2 = v2

    def parse(self, program):
        labels = {}
        lines = program.split("\n")
        self.lines = lines

        deprecated_instructions = set()
        instructions = []
        for i, line in enumerate(lines):
            label, mnemonic, arglist = self._parseline(line)
            if label:
                icnt = len(instructions)
                labels[label] = icnt
            if not mnemonic:
                continue

            if self.v2:
                if (mnemonic in ["jlt", "loop", "acquire_weighed"]
                        or (mnemonic == "jge" and len(arglist) == 3)):
                    deprecated_instructions.add(mnemonic)
                    mnemonic = "depr_" + mnemonic

            instructions.append(Instruction(i, mnemonic, arglist, label))

        if deprecated_instructions:
            print(f"Warning: Q1ASM contains deprecated instructions {deprecated_instructions}")

        self.labels = labels

        mnemonic_args = mnemonic_args_v2 if self.v2 else mnemonic_args_v1

        for instr in instructions:
            mnemonic = instr.mnemonic
            func_name = "_" + mnemonic
            instr.func_name = func_name
            if mnemonic in mnemonic_args:
                try:
                    args, reg_args = self._evaluate_args(mnemonic_args[mnemonic], instr.arglist)
                    instr.args = args
                    instr.reg_args = reg_args
                    if reg_args:
                        # 1 cycle for every register. instr and 1st register are loaded in 1 cycle
                        instr.clock_ticks = len(reg_args)
                    else:
                        # 1 cycle to load instruction
                        instr.clock_ticks = 1
                except AsmSyntaxError as ex:
                    print(ex)
                    print(lines[instr.text_line_nr])
                    ex.args = ex.args + (f"line {instr.text_line_nr}: {lines[instr.text_line_nr]}", )
                    raise
                except Exception as ex:
                    print(ex)
                    print(lines[instr.text_line_nr])
                    ex.args = ex.args + (f"line {instr.text_line_nr}: {lines[instr.text_line_nr]}", )
                    raise
            else:
                instr.args = instr.arglist

        return lines, instructions

    def _parseline(self, line):
        org_line = line
        label_pattern = r"(\w+:)"
        instr_pattern = r"(\w+:)?\s*(\w+)\s*(.*)"
        if line.startswith("#Q1Sim:"):
            return self._parse_simcmd(line[7:])
        try:
            end = line.index("#")
            line = line[:end]
        except Exception:
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
                arglist = args.replace(" ", "").split(",")
            else:
                arglist = []
            return [label, match.group(2), arglist]
        raise Exception(f"{self.name}: Parse error on line: {org_line}")

    def _parse_simcmd(self, command):
        command = command.strip()
        # format: log "msg",register,options
        log_pattern = r'log "(.*)",(\w+)?,(\w+)?'
        match = re.fullmatch(log_pattern, command)
        if match:
            msg = match.group(1)
            register = match.group(2)
            options = match.group(3)
            if msg is None:
                msg = ""
            return None, "log", (msg, register, options)
        trigger_pattern = r"sim_trigger (\d),\s*([01])"
        match = re.fullmatch(trigger_pattern, command)
        if match:
            addr = match.group(1)
            value = match.group(2)
            return None, "sim_trigger", (addr, value)
        print(f"Unknown simulator command:{command}")
        return None, None, None

    def _evaluate_args(self, arg_types, args):
        types = arg_types.split(",") if arg_types else []
        args = list(args)
        if len(args) != len(types):
            if len(args) > len(types):
                raise AsmSyntaxError(f"Incorrect number of arguments {len(args)}<>{len(types)}")
            # remaining types should be optional.
            for t in types[len(args):]:
                if t[-1] != "?":
                    raise AsmSyntaxError(f"Incorrect number of arguments {len(args)}<>{len(types)}")
        select_imm = False
        allow_imm = True
        arithmic_immediate_count = 0
        reg_args = []
        for i, arg in enumerate(args):
            allowed = types[i]
            c = arg[0]
            if allowed == "D":
                if c != "R":
                    raise AsmSyntaxError("Destination must be register")
                args[i] = self._parse_reg_arg(arg)
            elif c == "@":
                if "L" not in allowed:
                    raise AsmSyntaxError(f"Label operand not support as argument {i}")
                if "R" in allowed:
                    select_imm = True
                args[i] = self._parse_label_arg(arg)
            elif c == "R":
                if "R" in allowed:
                    if allowed[0] in "UuSsIAb":
                        # no immediate allowed where this is optional for this instruction
                        allow_imm = False
                elif "r" in allowed:
                    pass
                else:
                    raise AsmSyntaxError(f"Register operand not support as argument {i}")

                args[i] = self._parse_reg_arg(arg)
                reg_args.append(i)
            else:
                # parse immediate (integer)
                match allowed[0]:
                    case "I":
                        args[i] = self._parse_integer_arg(arg)
                    case "b":
                        args[i] = self._parse_bool_arg(arg)
                    case "A":
                        args[i] = self._parse_address_arg(arg)
                    case "U":
                        args[i] = self._parse_uint32_arg(arg)
                    case "u":
                        args[i] = self._parse_uint16_arg(arg)
                    case "S":
                        args[i] = self._parse_int32_arg(arg)
                    case "s":
                        args[i] = self._parse_int16_arg(arg)
                    case _:
                        raise AsmSyntaxError(f"Immediate operand not support as argument {i}")

                if "R" in allowed:
                    # no register allowed where this is optional for this instruction
                    select_imm = True
                elif "r" in allowed:
                    arithmic_immediate_count += 1

        if not allow_imm and select_imm:
            raise AsmSyntaxError("Combination of operands not supported")
        if arithmic_immediate_count > 1:
            raise AsmSyntaxError("Combination of operands not supported")

        if len(reg_args) == 0:
            reg_args = None
        return args, reg_args

    def _parse_reg_arg(self, arg):
        try:
            reg_nr = int(arg[1:])
        except ValueError:
            raise AsmSyntaxError(f"Invalid register '{arg}'") from None
        if reg_nr < 0 or reg_nr > 63:
            raise AsmSyntaxError(f"Invalid register '{arg}'")
        return reg_nr

    def _parse_label_arg(self, arg):
        try:
            line_nr = self.labels[arg[1:]]
        except KeyError:
            raise AsmSyntaxError(f"Label {arg} not defined") from None
        return line_nr

    def _parse_address_arg(self, arg):
        try:
            res = np.uint16(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid address: {arg}") from None
        if res >= 2**14:
            raise AsmSyntaxError(f"Invalid address: {arg}")
        return res

    def _parse_bool_arg(self, arg):
        try:
            res = np.uint16(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid bool: {arg}") from None
        if res >= 2:
            raise AsmSyntaxError(f"Invalid bool: {arg}")
        return res

    def _parse_integer_arg(self, arg):
        try:
            res = np.int64(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid integer: {arg}") from None
        if res >= 2**32 or res < -2**31:
            raise AsmSyntaxError(f"Invalid integer: {arg}")
        return np.uint32(res)

    def _parse_uint32_arg(self, arg):
        try:
            res = np.uint32(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid unsigned integer: {arg}") from None
        return res

    def _parse_int32_arg(self, arg):
        try:
            res = np.int32(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid integer: {arg}") from None
        return np.uint32(res)

    def _parse_uint16_arg(self, arg):
        try:
            res = np.uint16(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid unsigned integer: {arg}") from None
        return np.uint32(res)

    def _parse_int16_arg(self, arg):
        try:
            res = np.int16(arg)
        except (OverflowError, ValueError):
            raise AsmSyntaxError(f"Invalid integer: {arg}") from None
        return np.uint32(res)
