from collections import deque
import time
import logging

from .q1parser import Q1Parser

logger = logging.getLogger(__name__)

class Halt(Exception):
    pass

class Abort(Exception):
    pass

class Illegal(Exception):
    pass


class Q1Core:
    def __init__(self, name, renderer, is_qrm):
        self.name = name
        self.renderer = renderer
        self._is_qrm = is_qrm
        self.max_core_cycles= 10_000_000
        self.render_repetitions = True
        self.R = [0]*64
        self.lines = []
        self.instructions = []
        self.iptr = 0
        self.errors = set()

    def load(self, program):
        parser = Q1Parser()
        self.lines,self.instructions = parser.parse(program)

    def get_used_triggers(self):
        '''
        Returns int with OR of all used condition masks
        '''
        res = 0
        for instr in self.instructions:
            if instr.mnemonic == 'set_cond':
                mask = instr.args[1]
                if instr.reg_args and 1 in instr.reg_args:
                    # Trigger addresses are determined at run time.
                    logger.info('Condition mask is register.')
                    continue
                res |= mask
        return res

    def run(self):
        self.errors = set()
        self.R = [0]*64
        self.iptr = 0
        self.clock = CoreClock()
        # give the core a head start of 10 cycles
        self.clock.add_ticks(-10)

        if len(self.instructions) == 0:
            print(f'*** No instructions loaded')
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
        try:
            cntr = 0
            while(True):
                cntr += 1
                instr = self.instructions[self.iptr]
                self.iptr += 1
                if instr.reg_args is not None:
                    args = instr.args.copy()
                    for i in instr.reg_args:
                        args[i] = self.R[args[i]]
                else:
                    args = instr.args
                self.clock.add_ticks(instr.clock_ticks)
                instr.func(*args)
                if self.iptr >= len(self.instructions):
                    raise Illegal(f'No instruction at {self.iptr:04}')
                if cntr >= self.max_core_cycles:
                    raise Abort('Core cycle limited exceeded',
                                'FORCED STOP')
        except Halt:
            rt_time_us = self.renderer.time / 1000
            logger.info(f'{self.name}: stopped ({cntr} cycles, {rt_time_us:7.3f} us)')
        except Illegal as ex:
            msg = f'Illegal instruction at line {self.iptr}: {ex}'
            self._print_error_msg(msg, instr, cntr)
            self._error('SEQUENCE PROCESSOR Q1 ILLEGAL INSTRUCTION')
        except Abort as ex:
            msg = f'Execution aborted: {ex.args[0]}'
            self._print_error_msg(msg, instr, cntr)
            self._error(ex.args[1])
        except:
            self._print_error_msg('Exception', instr, cntr)
            self._error('OOPS!!')
            raise

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

    def _illegal(self):
        raise Illegal('illegal instruction')

    def _stop(self):
        raise Halt('stop instruction')

    def _nop(self):
        pass

    def _jmp(self, label):
        # 3 cycles for jump
        self.clock.add_ticks(3)
        self.iptr = label

    def _jlt(self, value, n, label):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        if value < n:
            # 3 cycles for jump
            self.clock.add_ticks(3)
            self.iptr = label

    def _jge(self, value, n, label):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        if value >= n:
            # 3 cycles for jump
            self.clock.add_ticks(3)
            self.iptr = label

    def _loop(self, register, label):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(register, self.R[register] - 1)
        instr = self.instructions[self.iptr-1]
        if not self.render_repetitions and instr.arglist[1] == '@_start':
            logger.info('Skipping repetitions')
            return
        if self.R[register] != 0:
            # 3 cycles for jump
            self.clock.add_ticks(3)
            self.iptr = label

    def _move(self, source, destination):
        self._set_register(destination, source)

    def _not(self, source, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, ~source)

    def _add(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs + rhs)

    def _sub(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs - rhs)

    def _and(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs & rhs)

    def _or(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs | rhs)

    def _xor(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs ^ rhs)

    def _asl(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs << rhs)

    def _asr(self, lhs, rhs, destination):
        # 2 cycles for arithmetic
        self.clock.add_ticks(2)
        self._set_register(destination, lhs >> rhs)

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
        self.renderer.set_awg_gain(gain0, gain1)

    def _set_awg_offs(self, offset0, offset1):
        self.renderer.set_awg_offs(offset0, offset1)

    def _set_cond(self, enable, mask, op, else_wait):
        self.renderer.set_cond(enable, mask, op, else_wait)

    def _upd_param(self, wait_after):
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.upd_param(wait_after)

    def _play(self, wave0, wave1, wait_after):
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.play(wave0, wave1, wait_after)

    def _acquire(self, bins, bin_index, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire(bins, bin_index, wait_after)

    def _acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire_weighed(bins, bin_index, weight0, weight1, wait_after)

    def _set_latch_en(self, enable, wait_after):
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.set_latch_en(enable, wait_after)

    def _latch_rst(self, wait):
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.latch_rst(wait)

    def _wait(self, time):
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.wait(time)

    def _wait_sync(self, wait_after):
        # assume wait_sync pauses the RT exec for at least 200 ns = 50 ticks
        self.clock.add_ticks(-50)
        self.renderer.wait_sync(wait_after)

    def _wait_trigger(self, wait_after):
        raise NotImplementedError()

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
        except: pass

        # q1core halts when buffer is full
        if len(b) >= 16:
            # q1core will continue when an instruction is read from buffer.
            # When q1core continues the time advantage is `time` - popped time.
            # So, core time will be equal to popped time
            self.core_time = b.popleft()

        b.append(time)
