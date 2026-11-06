"""
Pseudo random number generator in Q1ASM (xorshift with arithmetic shifts).

The random numbers are output as AWG offset. The simulated output is
compared with a Python reference model of the same 32-bit arithmetic.
"""
import itertools
import logging

import numpy as np
import pytest

from q1simulator import Q1Simulator


SEED = 820151979
N_LOOPS = 20
T_STEP = 400  # ns per random number
T_START = 16  # ns from wait_sync till first offset: upd_param 4 + 8 + 4
V_MAX = 2.5  # QCM

PROGRAM = f"""
            wait_sync    4                    #Wait for synchronization
            reset_ph                          #Reset absolute phase
            upd_param    4                    #Update all parameters and wait 4ns
start:
            move         {N_LOOPS},R1         #Init register R1: n loops
            move         {SEED},R61           #Init register R61: seed
            move         0,R62                #Init register R62: offset
            set_mrk      15                   #turn on all markers for triggering scope
            upd_param    8
            set_mrk      0                    #turn off all markers
            upd_param    4
program:
            set_awg_offs R62,R62              #set_awg gain for path0 = R62 and path1 = R62
            upd_param    {T_STEP}
            move         @ret1,R63
            jmp          @rng                 #jump to instructions
ret1:
            sub          R1,1,R1
            jnz          @program
stop:
            stop                     #stop sequencer:
rng:
            asl          R61, 13, R62
            nop
            xor          R61, R62, R61
            nop
            asr          R61, 17, R62
            nop
            xor          R61, R62, R61
            nop
            asl          R61, 5, R62
            nop
            xor          R61, R62, R61
            nop
            asr          R61, 19, R62 # Bit masking 16 bits
            nop
            jmp          R63                  #jump back to instructions
"""

MASK32 = (1 << 32) - 1
_name_counter = itertools.count()


def asl(x: int, n: int) -> int:
    return (x << n) & MASK32


def asr(x: int, n: int) -> int:
    signed = x - (1 << 32) if x & (1 << 31) else x
    return (signed >> n) & MASK32


def to_int16(x: int) -> int:
    x &= 0xFFFF
    return x - (1 << 16) if x & (1 << 15) else x


def expected_offsets() -> list[int]:
    """Offsets set by the program: the initial 0 followed by N_LOOPS-1 random numbers."""
    x = SEED
    offsets = [0]
    for _ in range(N_LOOPS - 1):
        x ^= asl(x, 13)
        x ^= asr(x, 17)
        x ^= asl(x, 5)
        offsets.append(to_int16(asr(x, 19)))
    return offsets


@pytest.fixture(scope="module")
def output():
    logging.disable(logging.CRITICAL)
    sim = Q1Simulator(f"rng{next(_name_counter)}", "QCM", isa_version=(2, 0))
    try:
        seq = sim.sequencer0
        seq.connect_out0("I")
        seq.sync_en(True)
        seq.sequence({"program": PROGRAM, "waveforms": {}, "weights": {}, "acquisitions": {}})
        sim.arm_sequencer(0)
        sim.start_sequencer(0)
        status = sim.get_sequencer_status(0, timeout=1)
        assert status.err_flags == []
        yield seq.get_output()
    finally:
        sim.stop_sequencer()
        logging.disable(logging.NOTSET)


def test_reference_model_is_random():
    # Guards the reference model itself: 13-bit signed values that differ.
    offsets = expected_offsets()[1:]
    assert all(-4096 <= v < 4096 for v in offsets)
    assert len(set(offsets)) == len(offsets)


def test_output_duration(output):
    data = output["I"]
    assert data.t_min == 0
    assert data.t_max == T_START + N_LOOPS * T_STEP
    assert len(data.data) == data.t_max


def test_output_is_random_offsets(output):
    expected = np.zeros(T_START + N_LOOPS * T_STEP)
    for i, offset in enumerate(expected_offsets()):
        t = T_START + i * T_STEP
        expected[t:t + T_STEP] = offset * V_MAX / 2**15

    np.testing.assert_array_equal(output["I"].data, expected)


@pytest.mark.parametrize("marker", ["M1", "M2", "M3", "M4"])
def test_marker_pulse(output, marker):
    points = [(int(t), int(v)) for t, v in output[marker].points]
    t_end = T_START + N_LOOPS * T_STEP
    # set_mrk 15 at 4 ns (after upd_param 4), set_mrk 0 at 12 ns (after upd_param 8).
    assert points == [(0, 0), (4, 0), (4, 1), (12, 1), (12, 0), (t_end, 0)]
