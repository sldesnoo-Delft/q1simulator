# Q1Simulator tests

All tests in this directory are pytest tests. They run on the simulator.
Tests that only check registers can also run on a Qblox cluster.

## Running the tests

```shell
pytest test/                                   # simulator only
pytest test/ --hw-ip 192.168.0.2               # also on hardware
pytest test/ --hw-ip 192.168.0.2 --hw-slot 4 --hw-qcm-slot 2 --hw-qrm-slot 6
```

| Option | Default | Description |
|---|---|---|
| `--hw-ip` | none | IP address of the Qblox cluster. Hardware tests are skipped if not given. |
| `--hw-slot` | 2 | Slot of the module used for single module hardware tests. |
| `--hw-qcm-slot` | 2 | Slot of the QCM for hardware tests with a QCM and a QRM. |
| `--hw-qrm-slot` | 6 | Slot of the QRM for hardware tests with a QCM and a QRM. |

Without `--hw-ip` the hardware variants are reported as skipped.

## Test files

| File | Runs on | What it checks |
|---|---|---|
| `test_alu.py` | sim + HW (`--hw-slot`) | 83 cases of add, sub, cmp, 16- and 32-bit multiplies (including the `l`/`h` variants), and/or/xor and shifts. Checks the result registers and the outcome of all 14 conditional jumps against a Python model. |
| `test_ints.py` | sim + HW (`--hw-slot`) | Flags of `cmp` at the 32-bit sign and carry boundaries via all conditional jumps. On the simulator also `get_alu_flags()`. |
| `test_feedback1.py` | sim + HW (`--hw-qcm-slot`, `--hw-qrm-slot`) | Feedback events: send/receive, two events at the same time, write combine, and fast repetition of 10 events. |
| `test_conditional.py` | sim | `get_output()` of if/else and if/elif/else playback with `set_cond` on a trigger set with `#Q1Sim:sim_trigger`. |
| `test_conditional2.py` | sim | `get_output()` of conditional playback on a QCM using a thresholded acquisition trigger of a QRM. Three scenarios, each with and without trigger. |
| `test_trigger_distribution.py` | sim | `get_output()` of 3 sequencers that emit and use triggers, for 6 setups with trigger value 0 and 1. |
| `test_qcodes_param.py` | sim | `get_output()` with QCoDeS `gain_awg_path0` and `offset_awg_path0` combined with Q1ASM `set_awg_offs`. |
| `test_real_mode.py` | sim | `get_output()` in real mode: NCO modulated path 0 plus unmodulated path 1. |
| `test_rng.py` | sim | `get_output()` of a pseudo random number generator in Q1ASM against a Python reference model. |
| `test_regressions.py` | sim | Regression tests for fixed bugs. Known open issues are marked `xfail(strict=True)`. |

## Supporting files

- `conftest.py`: command line options and fixtures.
  - `backend`: cluster with 1 module. Runs the test on the simulator and, with `--hw-ip`, on hardware.
  - `qcm_qrm_backend`: cluster with a QCM and a QRM. Runs on the simulator and on hardware.
  - `sim_cluster`: factory for simulated clusters. Clusters are stopped and closed after the test.
- `q1_helpers.py`: shared helpers.
  - 32-bit ALU model with the expected outcome of all conditional jumps.
  - Program that stores the outcome of all conditional jumps in R10..R23.
  - `render_path`: expected output of a series of `play` instructions.

## Expected-failure tests

Tests marked `xfail(strict=True)` document known open issues.
When an issue is fixed the test passes unexpectedly and fails the run: remove the marker then.
`test_sequencers_started_together_sync_at_same_time` is non-strict, because its outcome depends on thread timing.

## Notes on expected behaviour

Some expected results document the current behaviour of the simulator and should be checked on hardware.

1. **Latched offset in a skipped branch** (`test_conditional2.py::test_if_else_offset`).
   When the NOR branch is skipped, its `set_awg_offs -16384` is still applied by the next `play`.
   `Renderer._clear_latched_settings` only discards phase and frequency settings, not offset or gain.
2. **Thresholds in `test_conditional2.py`.**
   The simulated acquisition value is the acquisition time in ms (< 0.2).
   Threshold −0.5 always triggers and threshold 0.9 never triggers.
   Every scenario runs with both thresholds, so both branches are covered.
3. **Condition evaluation with trigger** (`test_conditional2.py::test_condition_evaluation_trigger`).
   Only the first repetition is checked. The QRM and QCM loops have different lengths (548 ns and 544 ns),
   so the trigger shifts relative to `set_cond` in later repetitions.
4. **Real mode** (`test_real_mode.py`).
   Modulation scales both paths by 1/√2, including the unmodulated path 1.
5. **Rendering starts at `wait_sync`.**
   By default the simulator only renders output after `wait_sync`.
   `test_qcodes_param.py` has no `wait_sync` and sets `skip_wait_sync` to False.
6. **Loops to `_start` are not repeated.**
   The simulator skips loops to label `_start` by default, so `test_trigger_distribution.py` renders 1 repetition.
7. **Mutually dependent triggers** (`test_trigger_distribution.py`).
   Two sequencers that use each other's trigger run correctly.
   Older versions of the simulator raised "Sorting sequencers on triggers failed" for this setup.
8. **ALU flags.**
   `test_alu.py` checks the flags that the original script checked.
   C and O of the 32-bit multiplies are not checked (TODO: check documentation).
9. **Carry flag of subtraction.**
   C is the borrow: `jb` jumps when `a < b` unsigned.

## Changes compared to the original scripts

- `test_feedback1.py`, section 2: removed `fb_pull_data` from QCM sequencer 0.
  Event 20 is only routed to QCM sequencer 1, so sequencer 0 aborted with a feedback timeout.
- All tests check that every sequencer finishes without error flags.
- Plotting and trace output were removed.
- Register values are passed as unsigned 32-bit values.
