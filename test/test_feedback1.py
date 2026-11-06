"""
Feedback events between sequencers: fb_com_data, fb_pull_data, fb_pop_data and write combine.

Runs on the simulator and, with `--hw-ip`, on a Qblox cluster. See conftest.py.
"""
import pytest


ACQUISITIONS = {"acq0": {"num_bins": 1, "index": 0}}

N_REP = 1
HEAD = f"""
move {N_REP},R1
wait_sync 100
"""

TAIL = """
loop R1,@start
stop
"""


def run(backend, programs, routes, out_registers) -> dict[tuple[str, int], dict[str, int]]:
    """Runs programs on QCM and QRM sequencers and returns registers.

    Args:
        programs: dict with (module name "qcm" or "qrm", sequencer number) -> program.
        routes: dict with event_id -> list of (module name, sequencer number).
        out_registers: dict with (module name, sequencer number) -> registers to read.
    """
    cluster = backend.cluster
    modules = {"qcm": backend.qcm, "qrm": backend.qrm}
    cluster.reset()
    for module in modules.values():
        module.sequencers[0].connect_out0("I")
        module.sequencers[1].connect_out1("Q")
    for seq_nr in [0, 1]:
        seq = backend.qrm.sequencers[seq_nr]
        seq.thresholded_acq_trigger_en(False)
        seq.integration_length_acq(100)

    for (module_name, seq_nr), program in programs.items():
        module = modules[module_name]
        seq = module.sequencers[seq_nr]
        seq.sequence({
            "program": program,
            "waveforms": {},
            "weights": {},
            "acquisitions": ACQUISITIONS if module_name == "qrm" else {},
        })
        seq.sync_en(True)
        module.arm_sequencer(seq_nr)

    cluster.clear_router()
    for event_id, targets in routes.items():
        cluster.set_cmm_route(event_id, [modules[name].sequencers[seq_nr] for name, seq_nr in targets])

    cluster.start_sequencer()
    for module_name, seq_nr in programs:
        status = modules[module_name].sequencers[seq_nr].get_sequencer_status(timeout=1)
        assert status.err_flags == [], f"{module_name} sequencer {seq_nr}"

    result = {}
    for (module_name, seq_nr), registers in out_registers.items():
        seq = modules[module_name].sequencers[seq_nr]
        # WORKAROUND: read registers 1 by 1 for bug in QBI
        regs = {}
        for reg in registers:
            regs.update(seq.get_registers([reg]))
        result[(module_name, seq_nr)] = regs
    return result


def test_send_receive(qcm_qrm_backend):
    # 20 -> qcm0, qcm1
    # 21 -> qcm1
    # 22 -> qrm0
    qrm0_program = HEAD + """
start:
  acquire 0,0,100
  fb_com_data 20,40,400
  fb_com_data 21,41,400
  fb_com_data 22,42,400
  wait 800

  fb_pull_data R12,R13 # receive #22
""" + TAIL

    qcm0_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pull_data R10,R11 # receive #20
""" + TAIL

    qcm1_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pop_data 20,R11 # receive #20
  fb_pull_data R12,R13 # receive #21
""" + TAIL

    regs = run(
        qcm_qrm_backend,
        programs={("qrm", 0): qrm0_program, ("qcm", 0): qcm0_program, ("qcm", 1): qcm1_program},
        routes={20: [("qcm", 0), ("qcm", 1)], 21: [("qcm", 1)], 22: [("qrm", 0)]},
        out_registers={("qcm", 0): ["R10", "R11"], ("qcm", 1): ["R11", "R12", "R13"], ("qrm", 0): ["R12", "R13"]},
    )
    assert regs[("qcm", 0)] == {"R10": 20, "R11": 40}
    assert regs[("qcm", 1)] == {"R11": 40, "R12": 21, "R13": 41}
    assert regs[("qrm", 0)] == {"R12": 22, "R13": 42}


def test_two_events_at_same_time(qcm_qrm_backend):
    # TODO What happens on HW? Also test with larger shift: 32 bits. Check OR.
    # 20 -> qcm1
    # 21 -> qcm1
    qrm0_program = HEAD + """
start:
  acquire 0,0,100
  fb_com_data 20,40,400
  wait 800
""" + TAIL

    qcm0_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100
  fb_com_data 21,41,400

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000
""" + TAIL

    qcm1_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pull_data R10,R11 # receive #20
  fb_pull_data R12,R13 # receive #21
""" + TAIL

    regs = run(
        qcm_qrm_backend,
        programs={("qrm", 0): qrm0_program, ("qcm", 0): qcm0_program, ("qcm", 1): qcm1_program},
        routes={20: [("qcm", 1)], 21: [("qcm", 1)]},
        out_registers={("qcm", 1): ["R10", "R11", "R12", "R13"]},
    )
    assert regs[("qcm", 1)] == {"R10": 21, "R11": 41, "R12": 20, "R13": 40}


def test_write_combine(qcm_qrm_backend):
    # 20 -> qcm1: data 5 at bit 0 OR data 10 at bit 8
    qrm0_program = HEAD + """
start:
  acquire 0,0,96
  fb_com_cfg 1, 0, 2, 4
  fb_com_data 20,5,400
  wait 2000
""" + TAIL

    qcm0_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 96
  fb_com_cfg 1, 8, 2, 4
  fb_com_data 20,10,400

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000
""" + TAIL

    qcm1_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 1000

  fb_pull_data R10,R11 # receive #20
""" + TAIL

    regs = run(
        qcm_qrm_backend,
        programs={("qrm", 0): qrm0_program, ("qcm", 0): qcm0_program, ("qcm", 1): qcm1_program},
        routes={20: [("qcm", 1)], 21: [("qcm", 1)]},
        out_registers={("qcm", 1): ["R10", "R11", "R12", "R13"]},
    )
    assert regs[("qcm", 1)] == {"R10": 20, "R11": 5 | (10 << 8), "R12": 0, "R13": 0}


@pytest.mark.parametrize("n_events,t_wait", [(10, 100)])
def test_fast_event_repetition(qcm_qrm_backend, n_events, t_wait):
    # 20 -> qcm1
    qrm0_program = HEAD + """
start:
  acquire 0,0,100
"""
    for i in range(n_events):
        qrm0_program += f"""
  fb_com_data 20,{100+i},{t_wait}
"""
    qrm0_program += f"""
  wait {20000-n_events*t_wait}
""" + TAIL

    qcm0_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 20000
""" + TAIL

    qcm1_program = HEAD + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 1000
"""
    for i in range(n_events):
        qcm1_program += f"""
  fb_pop_data 20,R{10+i}
  wait {t_wait}
"""
    qcm1_program += f"""
  wait {20000-1000-n_events*t_wait}
""" + TAIL

    registers = [f"R{10+i}" for i in range(n_events)]
    regs = run(
        qcm_qrm_backend,
        programs={("qrm", 0): qrm0_program, ("qcm", 0): qcm0_program, ("qcm", 1): qcm1_program},
        routes={20: [("qcm", 1)], 21: [("qcm", 1)]},
        out_registers={("qcm", 1): registers},
    )
    assert regs[("qcm", 1)] == {reg: 100 + i for i, reg in enumerate(registers)}
