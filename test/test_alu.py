"""
Tests the ALU instructions: result registers and flags via the outcome of all conditional jumps.

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


def int16(value: int) -> int:
    value &= 0xFFFF
    return value - (1 << 16) if value >= (1 << 15) else value


def add_sub_flags(ua: int, ub: int, unsigned_result: int, signed_result: int) -> dict:
    result = to_uint32(unsigned_result)
    return {
        "Z": result == 0,
        "N": result >= (1 << 31),
        "C": not (0 <= unsigned_result < (1 << 32)),
        "O": not (-(1 << 31) <= signed_result < (1 << 31)),
    }


def logic_flags(result: int) -> dict:
    return {"Z": result == 0, "N": result >= (1 << 31), "C": False, "O": False}


def expected(operator: str, a: int, b: int) -> tuple[list[int], dict]:
    """Returns expected destination registers (unsigned 32-bit) and flags.

    Flags with value None are not checked.
    """
    ua, ub = to_uint32(a), to_uint32(b)
    sa, sb = to_int32(a), to_int32(b)

    if operator in ("add", "sub", "cmp"):
        if operator == "add":
            flags = add_sub_flags(ua, ub, ua + ub, sa + sb)
            result = ua + ub
        else:
            flags = add_sub_flags(ua, ub, ua - ub, sa - sb)
            result = ua - ub
        return ([] if operator == "cmp" else [to_uint32(result)]), flags

    if operator == "mulu16":
        result = (ua & 0xFFFF) * (ub & 0xFFFF)
        return [result], {"Z": result == 0, "N": None, "C": False, "O": None}
    if operator == "muls16":
        result = int16(a) * int16(b)
        return [to_uint32(result)], {"Z": result == 0, "N": result < 0, "C": None, "O": False}

    if operator.startswith(("mulu32", "muls32")):
        signed_product = sa * sb
        product = signed_product if operator.startswith("muls") else ua * ub
        low, high = to_uint32(product), to_uint32(product >> 32)
        part = operator[6:]
        results = {"": [low, high], "l": [low], "h": [high]}[part]
        # TODO check C and O in documentation.
        return results, {"Z": signed_product == 0, "N": signed_product < 0, "C": None, "O": None}

    if operator in ("and", "or", "xor"):
        result = {"and": ua & ub, "or": ua | ub, "xor": ua ^ ub}[operator]
        return [result], logic_flags(result)

    if operator in ("asl", "lsl"):
        result = to_uint32(ua << b)
        flags = logic_flags(result)
        flags["C"] = bool((ua >> (32 - b)) & 1)  # last bit shifted out
        flags["O"] = (result >> 31) != (ua >> 31)  # sign changed
        return [result], flags
    if operator in ("asr", "lsr"):
        result = to_uint32(sa >> b) if operator == "asr" else ua >> b
        flags = logic_flags(result)
        flags["C"] = bool((ua >> (b - 1)) & 1)  # last bit shifted out
        return [result], flags

    raise ValueError(f"Unknown operator {operator}")


BIG_POS = (1 << 31) - 5
BIG_NEG = -(1 << 31) + 4

ADD_SUB_CMP_OPERANDS = [
    (1, 2), (-1, 1), (1, -1), (-1, 10), (1, -10), (60_000, 32_000), (30_000, -32_000),
    (BIG_POS, BIG_POS), (BIG_NEG, -10), (BIG_POS, BIG_NEG), (BIG_NEG, BIG_NEG),
]

CASES = (
    [("add", a, b) for a, b in ADD_SUB_CMP_OPERANDS]
    + [("sub", a, b) for a, b in ADD_SUB_CMP_OPERANDS]
    + [("cmp", a, b) for a, b in ADD_SUB_CMP_OPERANDS]
    + [("muls16", a, b) for a, b in [
        (3, 4), (0, 4), (3, -4), (3, -30_000), (-30_000, -31_000), (-30_000, 31_000), (-30_000, -40_000)]]
    + [("mulu16", a, b) for a, b in [
        (3, 4), (0, 4), (3, 30_000), (3, -30_000), (-30_000, -31_000), (-30_000, 31_000), (-30_000, -40_000)]]
    + [("muls32", a, b) for a, b in [
        (3, 4), (0, 4), (3, -4), (3, -30_000), (-30_000, -31_000), (-30_000, 31_000), (-30_000, -40_000),
        (-300_000, -10_000_000)]]
    + [("mulu32", a, b) for a, b in [(3, 4), (0, 4), (30_000, 31_000), (300_000, 10_000_000)]]
    + [("muls32l", -30_000, 31_000), ("muls32l", 300_000, -10_000_000),
       ("muls32h", 30_000, -31_000), ("muls32h", -300_000, 10_000_000),
       ("muls32h", 3, 4),  # not zero, but 0 in high 32 bit
       ("muls32l", 0xFFFF0008, 0x4000_0000),  # negative, but 0 in low 32 bit.
       ("muls32l", 0xFFF00008, 0x0001_0000),  # negative, but > 0 in low 32 bit.
       ]
    + [("mulu32l", 30_000, 31_000), ("mulu32l", 300_000, 10_000_000),
       ("mulu32h", 30_000, 31_000), ("mulu32h", 300_000, 10_000_000)]
    + [(operator, 0x1100, 0x0101) for operator in ("and", "or", "xor")]
    + [("asl", 0x1100, 5), ("asl", 0xF000_1000, 2), ("asl", 0x7000_1000, 2),
       ("lsl", 0x1101, 3), ("lsl", 0xF001_1000, 2), ("asl", 0x8000_1000, 2)]
    + [("asr", 0x8110_0100, 5), ("asr", 0x1000_1000, 2),
       ("lsr", 0x1101_0001, 8), ("lsr", 0x1001_1000, 2)]
)


def case_id(case):
    operator, a, b = case
    return f"{operator}-{to_uint32(a):#x}-{to_uint32(b):#x}"


@pytest.mark.parametrize("operator,a,b", CASES, ids=[case_id(c) for c in CASES])
def test_alu(backend, operator, a, b):
    expected_results, flags = expected(operator, a, b)
    destinations = ["R3", "R4"][:len(expected_results)]
    body = f"{operator} R1,R2" + "".join(f",{reg}" for reg in destinations)

    regs = run_single_sequencer(
        backend,
        jump_test_program(body),
        registers={"R1": to_uint32(a), "R2": to_uint32(b)},
        out_registers=destinations + list(JUMP_REGISTERS.values()),
    )

    results = [to_uint32(regs[reg]) for reg in destinations]
    assert results == expected_results, f"results {[hex(r) for r in results]}"

    taken = jumps_taken(regs)
    expected_taken = expected_jumps(flags)
    assert {jump: taken[jump] for jump in expected_taken} == expected_taken, f"flags expected: {flags}"
