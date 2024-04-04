# Changelog
All notable changes to Q1Simulator will be documented in this file.

## \[0.11.7] - 2024-04-04

- Use sequencer label for plotting.
- Fixed exception during plotting when max_render_time is reached and max_render_time is a float.

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
