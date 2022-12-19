# Changelog
All notable changes to Q1Simulator will be documented in this file.

## \[0.8.0] - 2022-12-19

- Update to qblox_instruments v0.8
- Added set_freq
- Changed arguments set_ph, set_ph_delta
- Added version check

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
