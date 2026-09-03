from qblox_instruments import Cluster
from q1simulator import Cluster as SimCluster
import numpy as np
import operator as op


def py_to_uint32(value):
    return np.uint32(np.int64(value))


def py_to_int32(value):
    return np.int32(np.int64(value))


def run(cluster, program, waveforms={}, weights={}, acquisitions={},
        registers={}, out_registers=[]) -> dict[str, int]:

    qrm = cluster.module2
    # qrm.config("trace", True)
    seq = qrm.sequencers[0]
    seq.sync_en(True)
    seq.sequence({
        'program': program,
        'waveforms': waveforms,
        'weights': weights,
        'acquisitions': acquisitions})
    seq.set_registers(registers)
    qrm.arm_sequencer(0)

    cluster.start_sequencer()
    seq.get_sequencer_status(timeout=1.0)
    try:
        flags = seq.get_alu_flags()
        sflags = ""
        for flag in ["ZF", "NF", "CF", "OF"]:
            if flags[flag]:
                sflags += flag[0]
            else:
                sflags += " "
    except AttributeError:
        sflags = None

    # WORKAROUND: read registers 1 by 1 for bug in QBI
    regs = {}
    for reg in out_registers:
        regs.update(seq.get_registers([reg]))
    return {
        "Regs": regs, #seq.get_registers(out_registers),
        "Flags": sflags,
        }


# %%
def test_operator(operator: str, a: int, b: int, n_dest: int = 1) -> tuple[np.uint32, str]:
    cluster.reset()

    dest1 = ",R3" if n_dest > 0 else ""
    dest2 = ",R4" if n_dest > 1 else ""

    program = f"""
        wait_sync 10

        {operator} R1,R2{dest1}{dest2}
        """

    jumps = ["jz", "jnz", "jo", "jno", "js", "jns", "jg", "jge", "jl", "jle", "ja", "jae", "jb", "jbe"]
    for r, cj in enumerate(jumps, 10):
        program += f"""
        move -1,R{r}
        {cj} @_{cj}
        move 0,R{r}
        jmp @_n_{cj}
    _{cj}:
        move 1,R{r}
    _n_{cj}:
    """

    program += """
        stop 1
    """

    res = run(cluster, program,
              registers={"R1": a, "R2": b},
              out_registers=["R3", "R4"] + [f"R{i+10}" for i in range(len(jumps))])
    regs = res["Regs"]

    for i, cj in enumerate(jumps, 10):
        value = regs[f"R{i}"]
        assert value in [0, 1]

    flags = res["Flags"]
    sflags = ""

    jump_results = {}
    for r, cj in enumerate(jumps, 10):
        jump_results[cj] = regs[f"R{r}"]
    for cj, flag in [("jz", "Z"), ("js", "N"), ("jb", "C"), ("jo", "O")]:
        if jump_results[cj]:
            sflags += flag
        else:
            sflags += " "
    if flags is None:
        flags = sflags
    else:
        assert flags == sflags

    mask = (1 << 32) - 1
    value = np.uint32(regs["R3"] & mask)
    if n_dest == 2:
        value = value + (np.int64(regs["R4"]) << 32)

    return value, flags


def test_add(a: int, b: int):
    res, flags = test_operator("add", a, b)
    print(f"{a:6} + {b:6} = {res} ({flags})")
    ures = np.uint32(res)
    sres = np.int32(ures)
    sa = py_to_int32(a)
    sb = py_to_int32(b)
    ua = np.uint32(sa)
    ub = np.uint32(sb)
    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags

    assert (res == 0) == zf
    # unsigned
    assert ures == ua + ub
    # print(ua, ub, np.uint64(ua)+ub)
    assert (np.uint64(ua) + ub >= (1 << 32)) == cf
    # signed
    assert sres == sa + sb
    assert (sres < 0) == nf
    assert ((a + b >= (1 << 31)) or (a + b < -(1 << 31))) == of


def test_sub(a: int, b: int):
    res, flags = test_operator("sub", a, b)
    print(f"{a:6} - {b:6} = {res} ({flags})")
    ures = np.uint32(res)
    sres = np.int32(ures)
    sa = py_to_int32(a)
    sb = py_to_int32(b)
    ua = np.uint32(sa)
    ub = np.uint32(sb)
    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags

    assert (res == 0) == zf
    # unsigned
    assert ures == ua - ub
    # print(ua, ub, np.uint64(ua)+ub)
    assert (np.uint64(ua) - ub >= (1 << 32)) == cf
    # signed
    assert sres == sa - sb
    assert (sres < 0) == nf
    assert ((a - b >= (1 << 31)) or (a - b < -(1 << 31))) == of


def test_cmp(a: int, b: int):
    _, flags = test_operator("cmp", a, b, n_dest=0)
    mask = (1 << 32) - 1
    res = np.uint32(a & mask) - np.uint32(b & mask)
    print(f"{a:6} - {b:6} = {res} ({flags})")
    ures = np.uint32(res)
    sres = np.int32(ures)
    sa = py_to_int32(a)
    sb = py_to_int32(b)
    ua = np.uint32(sa)
    ub = np.uint32(sb)
    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags

    assert (res == 0) == zf
    # unsigned
    assert ures == ua - ub
    # print(ua, ub, np.uint64(ua)+ub)
    assert (np.uint64(ua) - ub >= (1 << 32)) == cf
    # signed
    assert sres == sa - sb
    assert (sres < 0) == nf
    assert ((a - b >= (1 << 31)) or (a - b < -(1 << 31))) == of


def test_mul16(a: int, b: int, signed: bool):
    operator = "muls16" if signed else "mulu16"
    res, flags = test_operator(operator, a, b)
    ures = np.uint32(res)
    sres = np.int32(ures)
    if signed:
        print(f"{a:6} * {b:6} = {sres} ({flags})")
    else:
        mask = (1 << 32) - 1
        print(f"{a & mask:6} * {b & mask:6} = {ures} ({flags})")
    sa = np.int16(py_to_int32(a))
    sb = np.int16(py_to_int32(b))
    ua = np.uint16(sa)
    ub = np.uint16(sb)
    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags

    assert (res == 0) == zf
    if signed:
        assert sres == np.int32(sa) * sb
        assert (sres < 0) == nf
        assert ((a * b >= (1 << 31)) or (a * b < -(1 << 31))) == of
    else:
        assert ures == np.uint32(ua) * ub
        assert (np.uint64(ua) * ub >= (1 << 32)) == cf


def test_mul32(a: int, b: int, signed: bool, hl: str = ""):
    operator = "muls32" if signed else "mulu32"
    if hl == "h":
        operator += "h"
    elif hl == "l":
        operator += "l"
    if not hl:
        res, flags = test_operator(operator, a, b, n_dest=2)
        ures = np.uint64(res)
        sres = np.int64(ures)
    else:
        res, flags = test_operator(operator, a, b)
        ures = np.uint32(res)
        sres = np.int32(ures)

    part = f"[{hl}]" if hl else ""
    if signed:
        print(f"{a:6} * {b:6} {part}= {sres} ({flags})")
    else:
        mask = (1 << 64) - 1
        print(f"{a & mask:6} * {b & mask:6} {part} = {ures} ({flags})")
    sa = py_to_int32(a)
    sb = py_to_int32(b)
    ua = np.uint32(sa)
    ub = np.uint32(sb)
    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags
    expected_signed = np.int64(sa) * sb
    expected_unsigned = np.uint64(ua) * ub
    assert (expected_signed == 0) == zf
    assert (expected_signed < 0) == nf
    if signed:
        if hl == "l":
            expected_signed = np.int32(expected_signed)
        elif hl == "h":
            expected_signed = expected_signed >> 32
        assert sres == expected_signed
        # TODO check documentation
        # assert ((a * b >= (1 << 63)) or (a * b < -(1 << 63))) == of
    else:
        if hl == "l":
            expected_unsigned = np.uint32(expected_unsigned)
        elif hl == "h":
            expected_unsigned = expected_unsigned >> 32
        assert ures == expected_unsigned
        # TODO check documentation
        # assert (np.uint64(ua) * ub >= (1 << 64)) == cf


def test_bitwise(a, b, operator):
    def lsr(a, b):
        # unsigned shift
        mask = (1 << 32) - 1
        return (a & mask) >> b

    def asr(a, b):
        # unsigned shift
        mask = (1 << 32) - 1
        return np.int32(np.uint32(a & mask)) >> b

    res, flags = test_operator(operator, a, b)
    ures = np.uint32(res)
    sres = np.int32(res)

    print(f"{a:08X} {operator:3} {b:08X} = {ures:08X} ({flags})")

    operators = {
        "and": op.__and__,
        "test": op.__and__,
        "or": op.__or__,
        "xor": op.__xor__,
        "asl": op.__lshift__,
        "asr": asr,
        "lsr": lsr,
        "lsl": op.__lshift__,
        }
    func = operators[operator]
    expected = np.uint32(np.int64(func(a, b)))
    assert ures == expected

    nf = "N" in flags
    zf = "Z" in flags
    cf = "C" in flags
    of = "O" in flags

    assert (ures == 0) == zf
    assert (sres < 0) == nf

    if operator in ["asl", "lsl"]:
        sa = py_to_int32(a)
        first_out = (sa >> 31) & 1
        last_out = (sa >> (32-b)) & 1
        sign_bit = (sa >> (31-b)) & 1
        assert cf == (last_out != 0)
        assert of == (sign_bit != first_out)
    elif operator in ["asr", "lsr"]:
        sa = py_to_int32(a)
        last_out = (sa >> (b-1)) & 1
        assert cf == (last_out != 0)
        assert of == 0
    else:
        assert cf == 0
        assert of == 0


# %%

use_simulator = True

if use_simulator:
    cluster = SimCluster('test', {2: 'QRM'}, isa_version=(2, 0))
else:
    cluster = Cluster('test', "192.168.0.2")

# %%

np.seterr(over="ignore")

big_pos = (1 << 31) - 5
big_neg = -(1 << 31) + 4

test_add(1, 2)
test_add(-1, 1)
test_add(1, -1)
test_add(-1, 10)
test_add(1, -10)
test_add(60_000, 32_000)
test_add(30_000, -32_000)
test_add(big_pos, big_pos)
test_add(big_neg, -10)
test_add(big_pos, big_neg)
test_add(big_neg, big_neg)

test_sub(1, 2)
test_sub(-1, 1)
test_sub(1, -1)
test_sub(-1, 10)
test_sub(1, -10)
test_sub(60_000, 32_000)
test_sub(30_000, -32_000)
test_sub(big_pos, big_pos)
test_sub(big_neg, -10)
test_sub(big_pos, big_neg)
test_sub(big_neg, big_neg)

test_cmp(1, 2)
test_cmp(-1, 1)
test_cmp(1, -1)
test_cmp(-1, 10)
test_cmp(1, -10)
test_cmp(60_000, 32_000)
test_cmp(30_000, -32_000)
test_cmp(big_pos, big_pos)
test_cmp(big_neg, -10)
test_cmp(big_pos, big_neg)
test_cmp(big_neg, big_neg)

test_mul16(3, 4, signed=True)
test_mul16(0, 4, signed=True)
test_mul16(3, -4, signed=True)
test_mul16(3, -30_000, signed=True)
test_mul16(-30_000, -31_000, signed=True)
test_mul16(-30_000, 31_000, signed=True)
test_mul16(-30_000, -40_000, signed=True)

test_mul16(3, 4, signed=False)
test_mul16(0, 4, signed=False)
test_mul16(3, 30_000, signed=False)
test_mul16(3, -30_000, signed=False)
test_mul16(-30_000, -31_000, signed=False)
test_mul16(-30_000, 31_000, signed=False)
test_mul16(-30_000, -40_000, signed=False)

test_mul32(3, 4, signed=True)
test_mul32(0, 4, signed=True)
test_mul32(3, -4, signed=True)
test_mul32(3, -30_000, signed=True)
test_mul32(-30_000, -31_000, signed=True)
test_mul32(-30_000, 31_000, signed=True)
test_mul32(-30_000, -40_000, signed=True)
test_mul32(-300_000, -10_000_000, signed=True)

test_mul32(3, 4, signed=False)
test_mul32(0, 4, signed=False)
test_mul32(30_000, 31_000, signed=False)
test_mul32(300_000, 10_000_000, signed=False)

# test low / high
test_mul32(-30_000, 31_000, signed=True, hl="l")
test_mul32(300_000, -10_000_000, signed=True, hl="l")
test_mul32(30_000, -31_000, signed=True, hl="h")
test_mul32(-300_000, 10_000_000, signed=True, hl="h")
test_mul32(3, 4, signed=True, hl="h")  # not zero, but 0 in high 32 bit
test_mul32(0xFFFF0008, 0x4000_0000, signed=True, hl="l")  # negative, but 0 in low 32 bit.
test_mul32(0xFFF00008, 0x0001_0000, signed=True, hl="l")  # negative, but > 0 in low 32 bit.

test_mul32(30_000, 31_000, signed=False, hl="l")
test_mul32(300_000, 10_000_000, signed=False, hl="l")
test_mul32(30_000, 31_000, signed=False, hl="h")
test_mul32(300_000, 10_000_000, signed=False, hl="h")

# test and, or, xor,
test_bitwise(0x1100, 0x0101, "and")
test_bitwise(0x1100, 0x0101, "or")
test_bitwise(0x1100, 0x0101, "xor")

# test asl, asr, lsr, lsl
test_bitwise(0x1100, 5, "asl")
test_bitwise(0xF000_1000, 2, "asl")
test_bitwise(0x7000_1000, 2, "asl")
test_bitwise(0x1101, 3, "lsl")
test_bitwise(0xF001_1000, 2, "lsl")
test_bitwise(0x8000_1000, 2, "asl")

test_bitwise(0x8110_0100, 5, "asr")
test_bitwise(0x1000_1000, 2, "asr")
test_bitwise(0x1101_0001, 8, "lsr")
test_bitwise(0x1001_1000, 2, "lsr")
