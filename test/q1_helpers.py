"""
Helpers shared by the tests: 32-bit ALU reference model and expected AWG output.
"""
import numpy as np

MASK32 = (1 << 32) - 1

# Conditional jumps and the register storing whether the jump was taken.
JUMPS = ["jz", "jnz", "jo", "jno", "js", "jns", "jg", "jge", "jl", "jle", "ja", "jae", "jb", "jbe"]
JUMP_REGISTERS = {jump: f"R{i}" for i, jump in enumerate(JUMPS, 10)}


def to_uint32(value: int) -> int:
    return int(value) & MASK32


def to_int32(value: int) -> int:
    value = to_uint32(value)
    return value - (1 << 32) if value >= (1 << 31) else value


def expected_jumps(flags: dict[str, bool | None]) -> dict[str, bool]:
    """Expected outcome of conditional jumps for given flags Z, N, C, O.

    A flag with value None is not specified. Jumps depending on it are not returned.
    """
    z, n, c, o = flags["Z"], flags["N"], flags["C"], flags["O"]
    conditions = {
        "jz": ([z], lambda: z),
        "jnz": ([z], lambda: not z),
        "jo": ([o], lambda: o),
        "jno": ([o], lambda: not o),
        "js": ([n], lambda: n),
        "jns": ([n], lambda: not n),
        "jg": ([z, n, o], lambda: not z and n == o),
        "jge": ([n, o], lambda: n == o),
        "jl": ([n, o], lambda: n != o),
        "jle": ([z, n, o], lambda: z or n != o),
        "ja": ([z, c], lambda: not z and not c),
        "jae": ([c], lambda: not c),
        "jb": ([c], lambda: c),
        "jbe": ([z, c], lambda: z or c),
    }
    return {
        jump: bool(condition())
        for jump, (used_flags, condition) in conditions.items()
        if None not in used_flags
    }


def jump_test_program(body: str, wait_sync: int = 100) -> str:
    """Program executing `body` and storing the outcome of all conditional jumps in R10..R23."""
    program = f"""
        wait_sync {wait_sync}
        wait 1000
        {body}
    """
    for jump, reg in JUMP_REGISTERS.items():
        program += f"""
        move -1,{reg}
        {jump} @_{jump}
        move 0,{reg}
        jmp @_n_{jump}
    _{jump}:
        move 1,{reg}
    _n_{jump}:
    """
    program += """
        stop 1
    """
    return program


def run_single_sequencer(backend, program: str, registers: dict[str, int],
                         out_registers: list[str]) -> dict[str, int]:
    """Runs program on sequencer 0 of backend module and returns registers."""
    backend.cluster.reset()
    seq = backend.module.sequencers[0]
    seq.sync_en(True)
    seq.sequence({"program": program, "waveforms": {}, "weights": {}, "acquisitions": {}})
    seq.set_registers(registers)
    backend.cluster.arm_sequencer(backend.slot, 0)
    backend.cluster.start_sequencer(backend.slot, 0)
    seq.get_sequencer_status(timeout=1)

    # WORKAROUND: read registers 1 by 1 for bug in QBI
    regs = {}
    for reg in out_registers:
        regs.update(seq.get_registers([reg]))
    return regs


def jumps_taken(regs: dict[str, int]) -> dict[str, bool]:
    taken = {}
    for jump, reg in JUMP_REGISTERS.items():
        assert regs[reg] in (0, 1), f"{jump}: register {reg} not written ({regs[reg]:#x})"
        taken[jump] = regs[reg] == 1
    return taken


def lsb(v_max: float) -> float:
    return v_max / 2**15


def render_path(duration: int, plays: list[tuple[int, list[float], float]]) -> np.ndarray:
    """Expected AWG path output, relative to full scale.

    Args:
        duration: length of output in ns.
        plays: list with (start time, waveform data, gain) of every play.
            A waveform plays till its end or till the next play.
    """
    out = np.zeros(duration)
    plays = sorted(plays, key=lambda p: p[0])
    for i, (t, data, gain) in enumerate(plays):
        t_next = plays[i + 1][0] if i + 1 < len(plays) else duration
        end = min(t + len(data), t_next, duration)
        out[t:end] = gain * np.asarray(data)[:end - t]
    return out
