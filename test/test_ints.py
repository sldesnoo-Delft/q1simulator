"""
Tests the ALU flags of `cmp` via the outcome of all conditional jumps.

Runs on the simulator and, with `--hw-ip`, on a Qblox cluster. See conftest.py.
"""
import pytest

from q1_helpers import (
    JUMP_REGISTERS,
    expected_jumps,
    jump_test_program,
    jumps_taken,
    run_single_sequencer,
    to_int32,
    to_uint32,
)


def expected_flags(a: int, b: int) -> dict[str, bool]:
    """Flags of 32-bit a - b."""
    ua, ub = to_uint32(a), to_uint32(b)
    sa, sb = to_int32(a), to_int32(b)
    result = to_uint32(ua - ub)
    return {
        "Z": result == 0,
        "N": result >= (1 << 31),
        "C": ua < ub,  # borrow
        "O": not (-(1 << 31) <= sa - sb < (1 << 31)),
    }


# Values shifted to the top byte to hit the 32-bit sign and carry boundaries.
CMP_CASES = [
    (2 << 24, 5 << 24),
    (-128 << 24, 1 << 24),
    (-3 << 24, 127 << 24),
    (127 << 24, -3 << 24),
    (5 << 24, 5 << 24),
    (0, 0),
    (0, 1),
    (-1, 0),
    (0x7FFF_FFFF, -1),
    (-(1 << 31), 1),
]


@pytest.mark.parametrize("a,b", CMP_CASES, ids=[f"{a:#x}-{b:#x}" for a, b in CMP_CASES])
def test_cmp_flags(backend, a, b):
    regs = run_single_sequencer(
        backend,
        jump_test_program("cmp R1,R2"),
        registers={"R1": to_uint32(a), "R2": to_uint32(b)},
        out_registers=list(JUMP_REGISTERS.values()),
    )
    flags = expected_flags(a, b)
    assert jumps_taken(regs) == expected_jumps(flags), f"flags expected: {flags}"

    if backend.is_simulator:
        seq = backend.module.sequencers[0]
        assert {k: bool(v) for k, v in seq.get_alu_flags().items()} == {
            "ZF": flags["Z"], "NF": flags["N"], "CF": flags["C"], "OF": flags["O"],
        }
