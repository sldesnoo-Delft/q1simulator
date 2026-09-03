from collections import deque
import time
import logging

import numpy as np

from .q1parser import Q1Parser
from .event_distributor import EventDistributor

logger = logging.getLogger(__name__)


RT_BUFFER_SIZE = 32


class Halt(Exception):
    def __init__(self, msg, exit_code):
        super().__init__(f"{exit_code}: {msg}")
        self.exit_code = exit_code


class Abort(Exception):
    pass


class Illegal(Exception):
    pass


def update_rt(func):
    def wrapper(self, *args, **kwargs):
        self.clock.schedule_rt(self.renderer.time)
        func(self, *args, **kwargs)
        rt_time = self.renderer.time
        if rt_time - self._last_rt_update > 1000:
            self.event_distributor.set_sequencer_time(self.name, self.renderer.time)
    return wrapper


class Q1Core:
    max_instructions_qcm = 16384
    max_instructions_qrm = 12288

    def __init__(self, name, renderer, is_qrm, event_distributor: EventDistributor,
                 isa_version: int = 1):
        self.name = name
        self.renderer = renderer
        self.event_distributor = event_distributor
        self._is_qrm = is_qrm
        self.max_core_cycles = 10_000_000
        self.skip_loops = ("_start", )
        self.R = np.zeros(64, dtype=np.uint32)
        self.lines = []
        self.instructions = []
        self.iptr = 0
        self.abort = False
        self._last_rt_update = 0
        self.exit_code = -1
        self.errors = set()
        self.v2 = isa_version == 2
        if self.v2:
            self._jge = self._jge_v2
            self._depr_acquire_weighed = self._acquire_weighted
        else:
            self._jge = self._jge_v1
            self._jlt = self._jlt_v1
            self._jge = self._jge_v1
            self._loop = self._loop_v1
            self._asr = self._lsr  # it does not extend the sign.
            self._acquire_weighed = self._acquire_weighted

    def load(self, program):
        parser = Q1Parser(self.v2)
        self.lines, self.instructions = parser.parse(program)
        max_instructions = Q1Core.max_instructions_qrm if self._is_qrm else Q1Core.max_instructions_qcm
        if len(self.lines) > max_instructions:
            raise Exception(f"Program too big for instruction memory. {len(self.lines)} > {max_instructions}.")

    def get_used_triggers(self) -> int:
        '''
        Returns int with OR of all used condition masks
        '''
        res = 0
        for instr in self.instructions:
            if instr.mnemonic == 'set_cond':
                mask = int(instr.args[1])
                if instr.reg_args and 1 in instr.reg_args:
                    # Trigger addresses are determined at run time.
                    logger.info('Condition mask is register.')
                    continue
                res |= mask
        return res

    def run(self):
        self.abort = False
        self._last_rt_update = 0
        self.errors = set()
        self.exit_code = -1
        self.iptr = 0
        self.zf = 0  # zero
        self.nf = 0  # negative
        self.cf = 0  # unsigned carry
        self.of = 0  # signed overflow
        self.updating_reg = None
        self.clock = CoreClock()
        # give the core a head start of 10 cycles
        self.clock.add_ticks(-10)

        if len(self.instructions) == 0:
            print('*** No instructions loaded')
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
            return

        for instr in self.instructions:
            try:
                instr.func = getattr(self, instr.func_name)
            except AttributeError as ex:
                msg = f'Illegal instruction at line {instr.text_line_nr}: {ex}'
                self._print_error_msg(msg, instr, 0)
                self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
                return

        start = time.perf_counter()
        orig_err_settings = np.seterr(over='ignore')
        try:
            cntr = 0
            while True:
                cntr += 1
                instr = self.instructions[self.iptr]
                self.iptr += 1
                if instr.reg_args is not None:
                    args = instr.args.copy()
                    for i in instr.reg_args:
                        if i == self.updating_reg:
                            raise Exception(f"Register R{i} cannot be read immediately after write.")
                        args[i] = self.R[args[i]]
                else:
                    args = instr.args
                self.updating_reg = None
                self.clock.add_ticks(instr.clock_ticks)
                instr.func(*args)
                if self.iptr >= len(self.instructions):
                    raise Illegal(f'No instruction at {self.iptr:04}')
                if cntr >= self.max_core_cycles:
                    raise Abort('Core cycle limited exceeded',
                                'FORCED STOP')
                if self.abort:
                    raise Abort('Stop requested', 'FORCED STOP')
        except Halt as ex:
            rt_time_us = self.renderer.time / 1000
            self.exit_code = ex.exit_code
            logger.info(f'{self.name}: stopped ({cntr} cycles, {rt_time_us:7.3f} us), exit_code: {self.exit_code}')
        except Illegal as ex:
            msg = f'Illegal instruction at line {self.iptr}: {ex}'
            self._print_error_msg(msg, instr, cntr)
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
        except Abort as ex:
            msg = f'Execution aborted: {ex.args[0]}'
            self._print_error_msg(msg, instr, cntr)
            self._error(ex.args[1])
        except Exception:
            self._print_error_msg('Exception', instr, cntr)
            self._error("UNKNOWN")
            raise
        finally:
            np.seterr(**orig_err_settings)

        duration = time.perf_counter() - start
        logger.info(f'Duration {duration*1000:5.1f} ms {cntr} instructions, {duration/cntr*1e6:4.1f} us/instr')

    def _print_error_msg(self, msg, instr, cntr):
        last_line = self.lines[instr.text_line_nr]
        rt_time_us = self.renderer.time / 1000
        print(f'*** {self.name}: {msg} ({cntr} cycles, {rt_time_us:7.3f} us)')
        print(f'*** Last instruction: {last_line}')

    def _error(self, msg):
        self.errors.add(msg)

    def _set_register(self, reg_nr, value):
        self.R[reg_nr] = value
        self.updating_reg = reg_nr
        # print(f'R{reg_nr} = {np.int32(np.uint32(self.R[reg_nr]))} ({self.R[reg_nr]:08X})')

    def set_registers(self, registers: dict[str, int]):
        for reg_name, value in registers.items():
            reg_nr = int(reg_name[1:])
            self.R[reg_nr] = np.int64(value)

    def get_registers(self, registers: list[str] | None) -> dict[str, int]:
        res: dict[str, int] = {}
        if registers is None:
            registers = [f"R{i}" for i in range(len(self.R))]
        for reg_name in registers:
            reg_nr = int(reg_name[1:])
            res[reg_name] = int(self.R[reg_nr])
        return res

    def print_registers(self, reg_nrs=None):
        if reg_nrs is None:
            reg_nrs = range(64)
        for reg_nr in reg_nrs:
            value = self.R[reg_nr]
            signed_value = np.asarray(value).astype(np.int32)
            float_value = signed_value / 2**31
            print(f'R{reg_nr:02}: {value:08X} {signed_value:11}  ({float_value:9.6f})')

    def _set_result_flags(self, res):
        self.zf = res == 0
        self.nf = (res >> 31) & 1
        self.of = 0
        self.cf = 0

    # === Below are Q1ASM opcode mnemonics with _ prefix.

    def _illegal(self):
        raise Illegal('illegal instruction')

    def _stop(self, exit_code: int = 0):
        self.event_distributor.stop_sequencer(self.name)
        raise Halt('stop instruction', exit_code=exit_code)

    def _nop(self):
        pass

    def _jmp(self, addr):
        instr = self.instructions[self.iptr-1]
        # address/label is last argument of instruction
        jump_label = instr.arglist[-1][1:]
        if jump_label in self.skip_loops:
            logger.info(f'Skipping loop on {jump_label}')
            return
        # 3 cycles for jump
        self.clock.add_ticks(3)
        self.iptr = addr

    def _jz(self, addr):
        if self.zf:
            self._jmp(addr)

    def _jnz(self, addr):
        if not self.zf:
            self._jmp(addr)

    def _jo(self, addr):
        if self.of:
            self._jmp(addr)

    def _jno(self, addr):
        if not self.of:
            self._jmp(addr)

    def _js(self, addr):
        if self.nf:
            self._jmp(addr)

    def _jns(self, addr):
        if not self.nf:
            self._jmp(addr)

    def _jg(self, addr):
        if not self.zf and (self.nf == self.of):
            self._jmp(addr)

    def _jge_v2(self, addr):
        if self.nf == self.of:
            self._jmp(addr)

    def _jl(self, addr):
        if self.nf != self.of:
            self._jmp(addr)

    def _jle(self, addr):
        if self.zf or (self.nf != self.of):
            self._jmp(addr)

    def _ja(self, addr):
        if not self.zf and not self.cf:
            self._jmp(addr)

    def _jae(self, addr):
        if not self.cf:
            self._jmp(addr)

    def _jb(self, addr):
        if self.cf:
            self._jmp(addr)

    def _jbe(self, addr):
        if self.zf or self.cf:
            self._jmp(addr)

    def _jlt_v1(self, value, n, addr):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        if value < n:
            self._jmp(addr)

    def _jge_v1(self, value, n, addr):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        if value >= n:
            self._jmp(addr)

    def _loop_v1(self, register, addr):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(register, self.R[register] - 1)
        if self.R[register] != 0:
            self._jmp(addr)

    def _depr_jlt(self, value, n, addr):
        self._cmp(value, n)
        # 1 cycle to load instruction
        self.clock.add_ticks(1)
        self._jb(addr)

    def _depr_jge(self, value, n, addr):
        self._cmp(value, n)
        # 1 cycle to load instruction
        self.clock.add_ticks(1)
        self._jae(addr)

    def _depr_loop(self, register, addr):
        # 1 cycle to load register value
        self.clock.add_ticks(1)
        value = self.R[register]
        self._sub(value, 1, register)
        # 1 cycle to load instruction
        self.clock.add_ticks(1)
        self.updating_reg = None
        self._jnz(addr)

    def _move(self, source, destination):
        self._set_register(destination, source)

    def _not(self, source, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = ~source
        self._set_result_flags(res)
        self._set_register(destination, res)

    def _add(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)

        res = lhs + rhs

        # signed / usigned 32 bit calculations
        ures = np.uint64(lhs) + rhs
        sres = np.int64(np.int32(lhs)) + np.int32(rhs)
        self.cf = ures >= (1 << 32)
        self.of = sres >= (1 << 31) or sres < -(1 << 31)
        self.zf = res == 0
        self.nf = np.int32(sres) < 0
        self._set_register(destination, res)

    def _sub(self, lhs, rhs, destination: int | None):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = lhs - rhs

        # signed / usigned 32 bit calculations
        ures = np.uint64(lhs) - rhs
        sres = np.int64(np.int32(lhs)) - np.int32(rhs)
        self.cf = ures >= (1 << 32)
        self.of = sres >= (1 << 31) or sres < -(1 << 31)
        self.zf = res == 0
        self.nf = np.int32(sres) < 0
        if destination is not None:
            self._set_register(destination, lhs - rhs)

    def _cmp(self, lhs, rhs):
        self._sub(lhs, rhs, None)

    def _mulu16(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        mask = (1 << 16) - 1
        res = np.uint32(lhs & mask) * np.uint32(rhs & mask)
        self._set_result_flags(res)
        self._set_register(destination, res)

    def _muls16(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        # some casting to get correct signs
        res = np.int32(np.int16(lhs)) * np.int16(rhs)
        self._set_result_flags(res)
        self._set_register(destination, res)

    def _mulu32(self, lhs, rhs, destination_low: int | None, destination_high: int | None):
        self.clock.add_ticks(3)
        res = np.uint64(lhs) * np.uint64(rhs)
        mask = (1 << 32) - 1
        hres = res >> 32
        lres = res & mask
        self.zf = res == 0
        self.nf = np.int64(res) < 0
        self.cf = 0
        self.of = 0
        if destination_low is not None:
            self.clock.add_ticks(1)
            self._set_register(destination_low, lres)
        if destination_high is not None:
            self.clock.add_ticks(1)
            self._set_register(destination_high, hres)

    def _mulu32l(self, lhs, rhs, destination_low):
        self._mulu32(lhs, rhs, destination_low, None)

    def _mulu32h(self, lhs, rhs, destination_high):
        self._mulu32(lhs, rhs, None, destination_high)

    def _muls32(self, lhs, rhs, destination_low: int | None, destination_high: int | None):
        self.clock.add_ticks(3)
        res = np.int64(np.int32(lhs)) * np.int32(rhs)
        self.zf = res == 0
        self.nf = res < 0
        self.cf = 0
        self.of = 0
        if destination_low is not None:
            mask = (1 << 32) - 1
            self.clock.add_ticks(1)
            self._set_register(destination_low, res & mask)
        if destination_high is not None:
            self.clock.add_ticks(1)
            self._set_register(destination_high, res >> 32)

    def _muls32l(self, lhs, rhs, destination_low):
        self._muls32(lhs, rhs, destination_low, None)

    def _muls32h(self, lhs, rhs, destination_high):
        self._muls32(lhs, rhs, None, destination_high)

    def _and(self, lhs, rhs, destination: int | None):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = lhs & rhs
        self._set_result_flags(res)
        if destination is not None:
            self._set_register(destination, res)

    def _test(self, lhs, rhs):
        self._and(lhs, rhs, None)

    def _or(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = lhs | rhs
        self._set_result_flags(res)
        self._set_register(destination, res)

    def _xor(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = lhs ^ rhs
        self._set_result_flags(res)
        self._set_register(destination, res)

    def _asl(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = np.uint64(lhs) << rhs
        res_sign = (res >> 31) & 1
        self.cf = (res >> 32) & 1
        self.of = res_sign != (lhs >> 31) & 1
        self.zf = res == 0
        self.nf = res_sign != 0
        self._set_register(destination, res)

    def _lsl(self, lhs, rhs, destination):
        self._asl(lhs, rhs, destination)

    def _asr(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = np.int32(lhs) >> rhs
        res_sign = (res >> 31) & 1
        # last shifted bit
        self.cf = ((lhs >> (rhs - 1)) & 1) if rhs > 0 else 0
        self.of = 0
        self.zf = res == 0
        self.nf = res_sign != 0
        self._set_register(destination, res)

    def _lsr(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        res = np.uint32(lhs) >> rhs
        res_sign = (res >> 31) & 1
        # last shifted bit
        self.cf = ((lhs >> (rhs - 1)) & 1) if rhs > 0 else 0
        self.of = 0
        self.zf = res == 0
        self.nf = res_sign != 0
        self._set_register(destination, res)

    def _set_mrk(self, value):
        self.renderer.set_mrk(value)

    def _set_freq(self, freq):
        self.renderer.set_freq(freq)

    def _reset_ph(self):
        self.renderer.reset_ph()

    def _set_ph(self, phase):
        self.renderer.set_ph(phase)

    def _set_ph_delta(self, phase_delta):
        self.renderer.set_ph_delta(phase_delta)

    def _set_awg_gain(self, gain0, gain1):
        # cast for unsigned register values
        self.renderer.set_awg_gain(np.int16(gain0), np.int16(gain1))

    def _set_awg_offs(self, offset0, offset1):
        # cast for unsigned register values
        self.renderer.set_awg_offs(np.int16(offset0), np.int16(offset1))

    def _set_cond(self, enable, mask, op, else_wait):
        self.renderer.set_cond(enable, mask, op, else_wait)

    @update_rt
    def _upd_param(self, wait_after):
        self.renderer.upd_param(wait_after)

    @update_rt
    def _play(self, wave0, wave1, wait_after):
        self.renderer.play(wave0, wave1, wait_after)

    @update_rt
    def _acquire(self, bins, bin_index, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.renderer.acquire(bins, bin_index, wait_after)

    @update_rt
    def _acquire_weighted(self, bins, bin_index, weight0, weight1, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.renderer.acquire_weighted(bins, bin_index, weight0, weight1, wait_after)

    @update_rt
    def _acquire_ttl(self, bins, bin_index, enable, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.renderer.acquire_ttl(bins, bin_index, enable, wait_after)

    @update_rt
    def _set_latch_en(self, enable, wait_after):
        if enable not in [0, 1]:
            raise ValueError('enable must be 0 or 1')
        self.renderer.set_latch_en(enable, wait_after)

    @update_rt
    def _latch_rst(self, wait_after):
        self.renderer.latch_rst(wait_after)

    @update_rt
    def _wait(self, time):
        self.renderer.wait(time)

    @update_rt
    def _wait_sync(self, wait_after):
        self.renderer.wait_sync(wait_after)

    def _wait_trigger(self, wait_after):
        raise NotImplementedError()



    @update_rt
    def _fb_acq_iq_id(self, event_id, wait_after):
        ...

    @update_rt
    def _fb_acq_iq_shift(self, rshift, wait_after):
        ...

    @update_rt
    def _fb_acq_tb_id(self, event_id, wait_after):
        ...

    @update_rt
    def _fb_acq_tb_cfg(self, write_combine, bit_pos, length, wait_after):
        ...

    @update_rt
    def _fb_acq_tb_valid(self, valid, wait_after):
        ...

    @update_rt
    def _fb_acq_tb_extra(self, valid, data, wait_after):
        raise NotImplementedError()

    @update_rt
    def _fb_acq_tb_mock(self, enable, valid, data, wait_after):
        ...

    @update_rt
    def _fb_com_data(self, event_id, data, wait_after):
        ...

    @update_rt
    def _fb_com_cfg(self, write_combine, bit_pos, length, wait_after):
        ...

    @update_rt
    def _fb_com_extra(self, valid, data, wait_after):
        raise NotImplementedError()

    def _fb_pop_data(self, event_id, destination):
        timeout = False
        while not timeout:
            try:
                value = self.fb_queue.pop(event_id)
                self._set_register(destination, value)
                return
            except EmptyQueue:
                ... # Loop till RT Underflow. @@@
                self.scheduler.wait_till(process_id, time)


    def _fb_pull_data(self, destination_id, destination_data):
        ...

    # ---- Simulator commands ----

    def _log(self, msg, reg, options):
        if 'R' in options and reg.startswith('R'):
            reg_nr = int(reg[1:])
            value = self.R[reg_nr]
            signed_value = np.asarray(value).astype(np.int32)
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

    def _sim_trigger(self, addr, value):
        self.renderer.sim_trigger(addr, value)


class CoreClock:
    def __init__(self):
        self.rt_buffer = deque()
        self.core_time = 0

    def add_ticks(self, value):
        self.core_time += value * 4

    def schedule_rt(self, time):
        # print(f'Sched {time:6} at {self.core_time:6}')
        core_time = self.core_time
        if time < core_time:
            # rt command is already in the past w.r.t. the q1core time
            print(f'*** Schedule {time:6} at {self.core_time:6} ***')
            raise Abort('Real time buffer underrun',
                        'SEQUENCE PROCESSOR RT EXEC COMMAND UNDERFLOW')
        b = self.rt_buffer
        try:
            # remove executed rt entries.
            while b[0] < core_time:
                b.popleft()
        except Exception:
            pass

        # q1core halts when buffer is full
        if len(b) >= RT_BUFFER_SIZE:
            # q1core will continue when an instruction is read from buffer.
            # When q1core continues the time advantage is `time` - popped time.
            # So, core time will be equal to popped time
            self.core_time = b.popleft()

        b.append(time)
