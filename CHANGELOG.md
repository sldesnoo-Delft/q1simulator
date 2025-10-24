# Changelog
All notable changes to Q1Simulator will be documented in this file.

## \[0.18.2] - 2025-10-24

- Fixed QRM-RF qcodes parameters.

## \[0.18.1] - 2025-10-23

- Added `acquire_ttl` and associated QCoDeS parameters.
- Added `update_sequence` to sequencer.
- Added scope acquisition.

## \[0.18.0] - 2025-10-08

- Updates for qblox-instruments v0.18.0
- Added argument `as_numpy` in `get_acquisitions`.
- Fix simulation runtime > 2^32 ns.

## \[0.17.1] - 2025-07-03

- Implemented 1 ns resolution.
- Dropped support for qblox-instruments < 0.12.0

## \[0.17.0] - 2025-06-11

- Update to qblox-instruments v0.17.0

## \[0.16.0] - 2025-05-09

- Update to qblox-instruments 0.16.0
- Added option to specify frequency to tender analogue output. Default changed to 4GSa/a (was 20 GSa/s)
- Use pyproject.toml instead of setup.py

## \[0.15.4] - 2025-02-21

- Removed unintended print statements.

## \[0.15.3] - 2025-02-21

- Fixed bug in plotting introduced in 0.15.2.

## \[0.15.2] - 2025-02-21

- Added Q1ProgramBrowser (initial version).

## \[0.15.1] - 2025-02-10

- Fixed `set_freq` in Q1ASM.

## \[0.15.0] - 2025-01-17

- Update to qblox-instruments 0.15.0

## \[0.14.7] - 2024-12-04

- Added missing file with filter data.

## \[0.14.6] - 2024-12-04

- Added simulated output after analogue filter to Q1Plotter.

## \[0.14.5] - 2024-10-25

- Added simulated output after analogue filter of QCM (approximation).

## \[0.14.4] - 2024-10-18

- Update to qcodes 0.49
- Improved parse exceptions

## \[0.14.3] - 2024-10-16

- Fixed bug in rendering

## \[0.14.2] - 2024-10-16

- Dropped support for qblox-instruments < v0.11
- Support Numpy 2.0+
- Invalidate parameter cache on reset()

## \[0.14.1] -- failed --

## \[0.14.0] - 2024-10-14

- Added compatibility with qblox-instruments v0.14.1
- Added Real-Time Pre-Distortion (RTP) API-only
- Added Automatic Mixer Calibration (AMC) API-only
- Fixed output amplitude when NCO is enabled.
- Require Python >= 3.10

## \[0.13.1] - 2024-10-10

- Fixed Cluster.reset()
- Implemented qcodes params 'nco_phase_offs', 'gain_awg_path0', 'offset_awg_path0'

## \[0.13.0] - 2024-06-05

- Changed deprecated get_XXX_state to get_XXX_status

## \[0.11.10] - 2024-04-19

- Added acq_trigger_value to Q1Plotter initializer.

## \[0.11.9] - 2024-04-11

- Cleanup and fixes of Q1Plotter.

## \[0.11.8] - 2024-04-11

- Added Q1Plotter which plots the output of a cluster based on its current program and settings.
- Added setting `acq_trigger_value` to simulate with all acquisitions sending a trigger (or not).
- Improved trigger distribution even allowing conditions on triggers that are never emitted.
- Set max output voltage of q1viewer to 2.5 V (equal to QCM)

## \[0.11.7] - 2024-04-04

- Use sequencer label for plotting.
- Fixed exception during plotting when max_render_time is reached and max_render_time is a float.
- Added ignore_triggers to q1viewer.
- Added t_min, t_max to plot functions

## \[0.11.6] - 2024-03-27

- Fixed sequencer.sequence(s) if s is Path.

## \[0.11.5] - 2024-03-27

- Fixed q1viewer for qblox-instruments 0.11+.
- q1viewer now by default ignores sequence repetition and wait_sync time.

## \[0.11.4] - 2024-03-21

- Fixed RF module connect in/out for qblox-instruments v0.11+.

## \[0.11.3] - 2024-03-18

- Added nco_prop_delay_comp.

## \[0.11.2] - 2023-12-07

- Added argument check to set_latch_en

## \[0.11.1] - 2023-10-09

- Fixed module parameter markerX_inv_en

## \[0.11.0] - 2023-09-06

- Added qblox-instruments v0.11.0 features.
- Dropped compatibility with qblox-instruments v0.8

Note: Time resolution is still 4 ns.


## \[0.9.4] - 2023-08-03

- Corrected executing timing according to published Q1ASM specification
- Skip wait time after wait_sync by default.

## \[0.9.3] - 2023-07-23

- Added led_brightness of qblox-instruments v0.10.0

Note: Time resolution is still 4 ns. So not yet compatible with v0.10.

## \[0.9.2] - 2023-07-17

- Fixed acquisition triggers.
- Fixed implementation of conditionals and real time counters.

## \[0.9.1] - 2023-07-10

- Added config option render_repetitions to render of repeated sequence only once.
- Only plot output of sequencer with sync_en==True.

## \[0.9.0] - 2023-03-13

- Added conditional execution and triggers
- API and Q1ASM in line with qblox-instruments v0.9.0

## \[0.8.3] - 2023-03-03

- Added rendering of markers

## \[0.8.2] - 2023-02-08

- Added QRM ADC calibration.
- Changed logging.info() to logger.info()
- Added Cluster and ClusterModule

## \[0.8.1] - 2022-12-22

- Changed reset_ph to better match Qblox firmware 0.3.0.

## \[0.8.0] - 2022-12-22

- Update to qblox_instruments v0.8
- Added set_freq
- Changed arguments set_ph, set_ph_delta
- Added version check

## \[0.7.1] - 2022-12-21

- Improved performance with ~50%
- Changed reset_ph, set_ph, and set_ph_delta to match Qblox hardware

## \[0.7.0] - 2022-12-02

- Aligned version with qblox-instruments version
- Added `delete_acquisition_data` to module
- Improved performance with 20...40%

## \[0.4.6] - 2022-11-07

### Added
- Added get_idn()
- Added tracing of rt statements

## \[0.4.4] - 2022-08-16

### Added
- Added state flag "SEQUENCE PROCESSOR RT EXEC COMMAND UNDERFLOW"

## \[0.4.3] - 2022-07-27

### Fixed
- Execution of sequence without wait_sync

## \[0.4.2] - 2022-07-21

### Added
- Allow set sequence with dict argument

## \[0.4.1] - 2022-07-19

### Added
- Added handling of mixer_corr_gain_ratio and mixer_corr_phase_offset_degree to sequencer
- Added set_acquisition_mock_data to sequencer

### Changed
- plot output in V instead of raw DAC values

## \[0.4.0] - 2022-06-29

### Added
- Supported types 'QCM-RF' and 'QRM-RF'
- added RF parameters (simulator only logs values)

### Deleted
- Removed support for qblox-instruments < v0.6

## \[0.3.0] - 2022-04-13

### Changed
- Update for API changes of qblox_instruments v0.6
   - Note: still backwards compatible with qblox_instruments v0.5
- Instrument type 'QCM' or 'QRM' must be specified

## \[0.2.1] - 2022-04-01

### Fixed
- Return averaged acquisition data

## \[0.2.0] - 2022-03-07

### Fixed
- Return correct system status
- Accumulate acquisition data

## \[0.1.0] - 2022-02-24
First labeled release. Start of dev branch and change logging.
