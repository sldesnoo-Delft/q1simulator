import time
import logging

from .q1parser import Q1Parser

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
        self.R = [0]*64
        self.lines = []
        self.instructions = []
        self.iptr = 0
        self.errors = set()

    def load(self, program):
        parser = Q1Parser()
        self.lines,self.instructions = parser.parse(program)

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
            instr.func = getattr(self, instr.func_name)

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
                        # add 1 clock tick for every register access
                        self.clock.add_ticks(1)
                else:
                    args = instr.args
                instr.func(*args)
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
        except:
            self._print_error_msg('Exception', instr, cntr)
            self._error('OOPS!!')
            raise

        duration = time.perf_counter() - start
        logging.info(f'Duration {duration*1000:5.1f} ms {cntr} instructions, {duration/cntr*1e6:4.1f} us/instr')


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
        self.clock.add_ticks(1)

    def _jmp(self, label):
        self.clock.add_ticks(4)
        self.iptr = label

    def _jlt(self, value, n, label):
        self.clock.add_ticks(4)
        if value < n:
            self.iptr = label

    def _jge(self, value, n, label):
        self.clock.add_ticks(4)
        if value >= n:
            self.iptr = label

    def _loop(self, register, label):
        self.clock.add_ticks(5)
        self._set_register(register, self.R[register] - 1)
        if self.R[register] != 0:
            self.iptr = label

    def _move(self, source, destination):
        self.clock.add_ticks(1)
        self._set_register(destination, source)

    def _not(self, source, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, ~source)

    def _add(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs + rhs)

    def _sub(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs - rhs)

    def _and(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs & rhs)

    def _or(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs | rhs)

    def _xor(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs ^ rhs)

    def _asl(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs << rhs)

    def _asr(self, lhs, rhs, destination):
        self.clock.add_ticks(2)
        self._set_register(destination, lhs >> rhs)

    def _set_mrk(self, value):
        self.clock.add_ticks(1)
        self.renderer.set_mrk(value)

    def _reset_ph(self):
        self.clock.add_ticks(1)
        self.renderer.reset_ph()

    def _set_ph(self, arg0, arg1, arg2):
        self.clock.add_ticks(1)
        self.renderer.set_ph(arg0, arg1, arg2)

    def _set_ph_delta(self, arg0, arg1, arg2):
        self.clock.add_ticks(1)
        self.renderer.set_ph_delta(arg0, arg1, arg2)

    def _set_awg_gain(self, gain0, gain1):
        self.clock.add_ticks(1)
        self.renderer.set_awg_gain(gain0, gain1)

    def _set_awg_offs(self, offset0, offset1):
        self.clock.add_ticks(1)
        self.renderer.set_awg_offs(offset0, offset1)

    def _upd_param(self, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.upd_param(wait_after)

    def _play(self, wave0, wave1, wait_after):
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.play(wave0, wave1, wait_after)

    def _acquire(self, bins, bin_index, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire(bins, bin_index, wait_after)

    def _acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        if not self._is_qrm:
            raise NotImplementedError('instrument type is not QRM')
        self.clock.add_ticks(1)
        self.clock.schedule_rt(self.renderer.time)
        self.renderer.acquire_weighed(bins, bin_index, weight0, weight1, wait_after)

    def _wait(self, time):
        self.clock.add_ticks(1)
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
                        'SEQUENCE PROCESSOR RT EXEC COMMAND UNDERFLOW')
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
