"""
Regression tests for Q1Simulator. Simulator only, no hardware needed.

Run with:
    pytest test/test_regressions.py

Tests marked `xfail(strict=True)` document known open issues. When an issue
is fixed the test will XPASS and fail the run: remove the marker then.
"""
import itertools
import logging
import time

import numpy as np
import pytest
from qblox_instruments import SequencerStatusFlags

from q1simulator import Cluster, Q1Simulator
from q1simulator.event_distributor import EventDistributor, SequencerQueue, TriggerEvent
from q1simulator.q1parser import AsmSyntaxError, Q1Parser


EMPTY = {"waveforms": {}, "weights": {}, "acquisitions": {}}
_name_counter = itertools.count()

# A program that runs into a real-time buffer underrun: too many short RT instructions.
UNDERRUN_PROGRAM = "\n".join(
    ["wait_sync 4"]
    + ["set_mrk 1\nupd_param 4\nset_mrk 0\nupd_param 4"] * 200
    + ["stop"]
)
# A program that needs the time of other sequencers to advance (latch processing).
LATCH_PROGRAM = "wait_sync 4\nset_latch_en 1,4\nwait 60000\nwait 60000\nlatch_rst 4\nstop"


@pytest.fixture(autouse=True)
def quiet_logging():
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _wait_until_stopped(sequencers, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(seq.q1core.running for seq in sequencers):
            return True
        time.sleep(0.05)
    return False


def _shutdown(stop_sequencers, sequencers):
    """Stops sequencers. Threads still running after this would block exit of pytest."""
    stop_sequencers()
    assert _wait_until_stopped(sequencers, timeout=5), "sequencer threads still running after stop_sequencer()"


@pytest.fixture
def sim():
    sim = Q1Simulator(f"sim{next(_name_counter)}", sim_type="QRM", isa_version=(2, 0))
    yield sim
    _shutdown(sim.stop_sequencer, sim.sequencers)


@pytest.fixture
def cluster():
    cluster = Cluster(f"cluster{next(_name_counter)}", {2: "QRM"}, isa_version=(2, 0))
    yield cluster
    _shutdown(cluster.stop_sequencer, cluster.module2.sequencers)
    cluster.close()


def load(seq, program, **sequence):
    seq.sync_en(True)
    seq.sequence({"program": program, **EMPTY, **sequence})


def run(seq, program, **sequence):
    """Uploads, arms and starts a single sequencer and returns its status."""
    load(seq, program, **sequence)
    seq.arm_sequencer()
    seq.start_sequencer()
    return seq.get_sequencer_status(timeout=1)


def start_together(*seqs):
    for seq in seqs:
        seq.arm_sequencer()
    for seq in seqs:
        seq.start_sequencer()


def routed_sequencers(distributor: EventDistributor, event_id: int) -> set[str]:
    return set(distributor._event_targets[event_id])


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_fb_acq_tb_valid():
    _, instructions = Q1Parser(True).parse("fb_acq_tb_valid 1,4\nstop")
    assert [int(a) for a in instructions[0].args] == [1, 4]


def test_parse_malformed_line_raises_syntax_error():
    with pytest.raises(AsmSyntaxError):
        Q1Parser(True).parse("nop\n-bad line\nstop")


def test_parse_unknown_instruction_raises_syntax_error():
    with pytest.raises(AsmSyntaxError):
        Q1Parser(True).parse("nop\nfoo R1\nstop")


def test_parse_unknown_simulator_command_raises_syntax_error():
    with pytest.raises(AsmSyntaxError):
        Q1Parser(True).parse("#Q1Sim: bogus\nstop")


def test_simulator_commands_take_no_clock_ticks():
    _, instructions = Q1Parser(True).parse('#Q1Sim: log "x",R0,R\nstop')
    log = instructions[0]
    assert log.is_simulator_instruction
    assert log.clock_ticks == 0


def test_comment_lines_do_not_count_toward_instruction_memory(sim):
    program = "\n".join(["# comment"] * 13000 + ["stop"])
    sim.sequencers[0].q1core.load(program)  # must not raise


@pytest.mark.parametrize("program", ["nop\n-bad line\nstop", "nop\nfoo R1\nstop"])
def test_parse_errors_include_line_number(program):
    with pytest.raises(AsmSyntaxError) as exc_info:
        Q1Parser(True).parse(program)
    assert "line 1" in str(exc_info.value)


@pytest.mark.xfail(strict=True, reason="#Q1Sim: lines are counted toward instruction memory")
def test_simulator_commands_do_not_count_toward_instruction_memory(sim):
    program = "\n".join(['#Q1Sim: log "x",,'] * 12300 + ["stop"])
    sim.sequencers[0].q1core.load(program)


# ---------------------------------------------------------------------------
# Q1Core execution and status
# ---------------------------------------------------------------------------

def test_log_between_write_and_read_does_not_hide_error(sim):
    status = run(sim.sequencers[0], 'move 1,R0\n#Q1Sim: log "x",R0,R\nadd R0,1,R1\nstop')
    assert status.err_flags, "read immediately after write must be reported"


def test_wait_too_short_reports_status_flag(sim):
    status = run(sim.sequencers[0], "wait 2\nstop")
    assert status.err_flags == [SequencerStatusFlags.SEQUENCE_PROCESSOR_RT_EXEC_ILLEGAL_INSTRUCTION]


def test_log_without_options(sim):
    status = run(sim.sequencers[0], '#Q1Sim: log "hello",,\nstop')
    assert status.err_flags == []


@pytest.mark.xfail(strict=True, reason="proposed: set_scope_en should be logged, not abort the run")
def test_set_scope_en_does_not_abort(sim):
    status = run(sim.sequencers[0], "set_scope_en 1\nupd_param 4\nstop")
    assert status.err_flags == []


# ---------------------------------------------------------------------------
# Multi-sequencer synchronization
# ---------------------------------------------------------------------------

def test_error_in_one_sequencer_does_not_hang_others(sim):
    a, b = sim.sequencers[0], sim.sequencers[1]
    load(a, UNDERRUN_PROGRAM)
    load(b, LATCH_PROGRAM)
    start_together(a, b)

    assert _wait_until_stopped([a, b], timeout=10), "sequencer hangs after other sequencer failed"
    assert "SEQUENCE PROCESSOR RT EXEC COMMAND UNDERFLOW" in a.q1core.errors
    assert b.q1core.errors == set()


def _start_with_blocked_wait_sync(sim):
    """Starts sequencer b with wait_sync and sync_en sequencer a without program."""
    a, b = sim.sequencers[0], sim.sequencers[1]
    a.sync_en(True)  # no program loaded
    load(b, LATCH_PROGRAM)
    start_together(a, b)
    return a, b


def test_sync_en_sequencer_without_program_blocks_wait_sync(sim):
    """Like the HW: a sync_en sequencer that never executes wait_sync blocks wait_sync of the others."""
    _, b = _start_with_blocked_wait_sync(sim)
    assert not _wait_until_stopped([b], timeout=1), "wait_sync should block on sequencer without program"


def test_stop_sequencer_releases_blocked_wait_sync(sim):
    a, b = _start_with_blocked_wait_sync(sim)
    assert not _wait_until_stopped([b], timeout=1), "wait_sync should block on sequencer without program"

    sim.stop_sequencer()
    assert _wait_until_stopped([a, b], timeout=5), "sequencers do not stop after stop_sequencer()"
    assert "FORCED STOP" in b.q1core.errors


def test_stop_sequencer_releases_wait_sync_after_rendering_started(sim):
    sim.config("skip_wait_sync", False)  # render from t=0
    a, b = sim.sequencers[0], sim.sequencers[1]
    a.sync_en(True)  # no program loaded
    load(b, "upd_param 1000\nwait_sync 4\nstop")
    start_together(a, b)
    assert not _wait_until_stopped([b], timeout=1), "wait_sync should block on sequencer without program"

    sim.stop_sequencer()
    assert _wait_until_stopped([a, b], timeout=5), "sequencers do not stop after stop_sequencer()"
    assert b.q1core.errors == {"FORCED STOP"}
    assert b.rt_renderer.time >= 1000, "time must not go back on aborted wait_sync"


def test_keyboard_interrupt_in_get_sequencer_status_does_not_stop_sequencer(sim, monkeypatch):
    """Like the HW: Ctrl-C in get_sequencer_status only interrupts the caller.
    The sequencer keeps running till stop_sequencer() is called.
    """
    a, b = _start_with_blocked_wait_sync(sim)
    assert not _wait_until_stopped([b], timeout=1), "wait_sync should block on sequencer without program"

    def interrupt(sequencer_name):
        raise KeyboardInterrupt()

    # Simulate Ctrl-C while get_sequencer_status polls the sequencer thread.
    monkeypatch.setattr(sim._scheduler, "join_sequencer", interrupt)
    with pytest.raises(KeyboardInterrupt):
        b.get_sequencer_status(timeout=1)
    monkeypatch.undo()
    assert not _wait_until_stopped([b], timeout=1), "sequencer should keep waiting after KeyboardInterrupt"

    sim.stop_sequencer()
    assert _wait_until_stopped([a, b], timeout=5), "sequencers do not stop after stop_sequencer()"
    assert b.q1core.errors == {"FORCED STOP"}

    # Next run is not affected.
    status = run(b, "wait 1000\nwait 1000\nstop")
    assert status.err_flags == []
    assert status.exit_code == 0
    assert b.get_simulation_end_time() == 2000


def test_stopped_wait_sync_does_not_affect_next_run(sim):
    # Run 1: b is stopped while blocked in wait_sync.
    a, b = _start_with_blocked_wait_sync(sim)
    assert not _wait_until_stopped([b], timeout=1), "wait_sync should block on sequencer without program"
    sim.stop_sequencer()
    assert _wait_until_stopped([a, b], timeout=5), "sequencers do not stop after stop_sequencer()"

    # Run 2: a arrives at wait_sync first and must wait for b.
    load(a, "wait_sync 4\nstop")
    load(b, "wait 65535\nwait_sync 4\nstop")
    start_together(a, b)

    assert _wait_until_stopped([a, b], timeout=5), "wait_sync hangs in run after stopped wait_sync"
    assert a.q1core.errors == set()
    assert b.q1core.errors == set()
    assert a.get_simulation_end_time() >= 65535, "a passed wait_sync without waiting for b"


def test_long_wait_sync_is_not_a_wait_time_error(sim):
    a, b = sim.sequencers[0], sim.sequencers[1]
    load(a, "wait 65535\nwait 65535\nwait_sync 4\nstop")
    load(b, "wait_sync 4\nstop")
    start_together(a, b)

    assert a.get_sequencer_status(timeout=1).err_flags == []
    assert b.get_sequencer_status(timeout=1).err_flags == []


@pytest.mark.xfail(strict=False, reason="start offset is taken from progress of already running sequencer threads")
def test_sequencers_started_together_sync_at_same_time(sim):
    a, b = sim.sequencers[0], sim.sequencers[1]
    load(a, "wait 65535\nwait 65535\nwait_sync 4\nstop")
    load(b, "wait_sync 4\nstop")
    sim.arm_sequencer(0)
    sim.arm_sequencer(1)
    sim.start_sequencer()
    a.get_sequencer_status(timeout=1)
    b.get_sequencer_status(timeout=1)

    # b has to wait in wait_sync until a arrives at 2*65535 ns.
    assert b.get_simulation_end_time() >= 2 * 65535


# ---------------------------------------------------------------------------
# Event distribution
# ---------------------------------------------------------------------------

def test_trigger_and_feedback_at_same_time_are_sorted():
    queue = SequencerQueue()
    queue.add_feedback_event(100, 3, [1], False)
    queue.add_trigger(100, 1, 1)
    # Trigger has the lower event type and is delivered first.
    assert isinstance(queue.get_event(100), TriggerEvent)


def test_self_cast_feedback_event():
    distributor = EventDistributor()
    distributor.arm_sequencer("seq", False)
    distributor.start_sequencer("seq")
    distributor.fb_send("seq", 100, 3, [1], "q1", False)

    event = distributor._sequencer_queue["seq"].get_event(10_000)
    assert event.event_id == 3
    assert event.values == [1]


# ---------------------------------------------------------------------------
# Routing API
# ---------------------------------------------------------------------------

def test_q1simulator_set_broad_cast(sim):
    sim.set_broad_cast(20)
    distributor = sim._scheduler.event_distributor
    assert routed_sequencers(distributor, 20) == {seq.name for seq in sim.sequencers}


def test_q1module_set_local_route(sim):
    sim.set_local_route(3)
    distributor = sim._scheduler.event_distributor
    assert routed_sequencers(distributor, 3) == {seq.name for seq in sim.sequencers}


def test_cluster_set_broad_cast(cluster):
    cluster.set_broad_cast(20)
    assert routed_sequencers(cluster._event_distributor, 20) == {
        seq.name for seq in cluster.module2.sequencers
    }


def test_clear_router_before_arm(sim):
    sim.set_local_route(3)
    sim.clear_router()
    assert routed_sequencers(sim._scheduler.event_distributor, 3) == set()


def test_cluster_delete_acquisition_data_uses_sequencer(cluster):
    seq = cluster.module2.sequencers[0]
    load(seq, "stop", acquisitions={"acq": {"index": 0, "num_bins": 2}})
    # Slot 2, sequencer 0. Used to look up sequencer 2 (the slot number).
    cluster.delete_acquisition_data(2, 0, name="acq")


# ---------------------------------------------------------------------------
# Rendering: gain and offset
# ---------------------------------------------------------------------------

def test_qcodes_and_q1asm_gain_are_multiplied(sim):
    seq = sim.sequencers[0]
    seq.connect_out0("I")
    seq.gain_awg_path0(0.5)
    waveforms = {"half": {"index": 0, "data": [0.5] * 100}}
    status = run(seq, "wait_sync 4\nset_awg_gain 16384,16384\nplay 0,0,100\nstop", waveforms=waveforms)
    assert status.err_flags == []

    data = seq.get_output()["I"].data
    v_max = 0.5  # QRM
    np.testing.assert_allclose(data[10:90], 0.5 * 0.5 * 0.5 * v_max, rtol=1e-3)


def test_qcodes_and_q1asm_offset_are_added(sim):
    seq = sim.sequencers[0]
    seq.connect_out0("I")
    seq.offset_awg_path0(0.1)
    status = run(seq, "wait_sync 4\nset_awg_offs 3277,0\nupd_param 100\nstop")
    assert status.err_flags == []

    data = seq.get_output()["I"].data
    v_max = 0.5  # QRM
    np.testing.assert_allclose(data[10:90], 0.2 * v_max, rtol=1e-3)


@pytest.mark.xfail(strict=True, reason="nco_phase_offs is written to next_settings, which reset() discards at run start")
def test_nco_phase_offs_is_applied(sim):
    seq = sim.sequencers[0]
    seq.connect_out0("I")
    seq.connect_out1("Q")
    seq.mod_en_awg(True)
    seq.nco_freq(0)
    seq.nco_phase_offs(90)
    seq.offset_awg_path0(0.5)
    status = run(seq, "wait_sync 4\nupd_param 100\nstop")
    assert status.err_flags == []

    output = seq.get_output()
    v_max = 0.5  # QRM
    # Constant I input rotated by 90 degrees: all signal on Q, scaled by 1/sqrt(2) for modulation.
    np.testing.assert_allclose(output["I"].data[10:90], 0.0, atol=1e-3)
    np.testing.assert_allclose(output["Q"].data[10:90], 0.5 * v_max / np.sqrt(2), rtol=1e-3)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def test_ttl_auto_bin_increment_beyond_num_bins(sim):
    seq = sim.sequencers[0]
    seq.ttl_acq_auto_bin_incr_en(True)
    status = run(
        seq,
        "wait_sync 4\nacquire_ttl 0,1,1,4\nwait 100\nacquire_ttl 0,1,0,4\nstop",
        acquisitions={"acq": {"index": 0, "num_bins": 2}},
    )
    assert SequencerStatusFlags.UNKNOWN not in status.err_flags
