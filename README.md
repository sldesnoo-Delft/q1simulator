# Q1Simulator
Simulator to execute Q1ASM and render the output channels.
The simulator can be used like a `Cluster` object or a single module
of `qblox_instruments`, where the Cluster can contain
QCM, QRM, QCM-RF and QRM-RF modules.
The simulator can be used with other Python software to test the
generation of Q1ASM programs without using the actual hardware.
It can also be used stand-alone to check and display Q1ASM
waveforms-and-program files.

Q1ASM program and waveforms can be uploaded to the sequencer.
When the sequencer is started it executes the Q1ASM instructions
and renders the output to a numpy array with DAC output values.

The `acquire` instructions store the timestamp of the real-time
executor in the path data and increments the avg_cnt.
This makes it easy to check the timing of the acquire instructions.
The acquire instructions also check the time between the triggers
and report an error when there would be an FIFO error.
The acquisition data returned by the sequencer can be set with
`set_acquisition_mock_data`.

The complete Q1ASM instruction set has been implemented.
Q1Simulator simulates the (estimated) execution time of Q1ASM and
uses buffer between Q1Core and real-time executor. It aborts execution
when the real-time buffer would have an underrun.
The renderer uses the uploaded waveforms and the nco frequency.

Note: Currently all qcodes parameters of the simulator initialize to None.
The parameters must explicitly be set.

# Example

```Python
    from q1simulator import Q1Simulator as Module

    sim = Module('q1sim', sim_type='QCM')
    sim.sequencer0.sequence('my_sequence.json')
    sim.arm_sequencer(0)
    sim.start_sequencer()
    sim.get_sequencer_state(0)
    sim.plot()
    sim.print_acquisitions()
```

# Cluster

A simulated Cluster can be created with the modules
specified in a dictionary with  slot number and module type.

```Python
    from q1simulator import Cluster
    modules = {
        2: "QCM",
        4: "QCM",
        6: "QRM",
        }
    cluster = Cluster('name', modules)

    qcm2 = cluster.module2
    qcm2.sequence('program.json')

    cluster.arm_sequencer()
    cluster.start_sequencer()
```

# Q1ProgramBrowser

Programs with configuration can be saved using Q1Pulse's `save_program` from `Q1Instrument`,
or setting `pulse_lib.qblox.QbloxConfig.store_programs=True` (for pulse-lib users).
Every program is stored in a new subdirectory.

```Python
    from q1pulse import Q1Instrument

    instrument = Q1Instrument(cluster)
    ...
    instrument.save_program("./stored_programs/program_xyz", program)
```

The saved programs can be browsed with `Q1ProgramBrowser`.
The browser shows the programs found in the subdirectories.

```Python
    from q1simulator import Q1ProgramBrowser
    Q1ProgramBrowser("./stored_programs")
```

# Q1Plotter

Q1Plotter retrieves the settings from a cluster and runs a simulation of
the program to plot the expected outputs.

```Python
    from q1simulator import Q1Plotter

    plotter = Q1Plotter(my_cluster)
    plotter.plot()
```
**NOTE: Only sequencers with `sync_en()==True` are copy to the simulator and plotted. **

# Q1ASM file viewer
`plot_q1asm_file` is a simple function that reads a file
and plots and prints the results.

```Python
    from q1simulator import plot_q1asm_file

    path = 'demo_data'
    plot_q1asm_file(path + r'/q1seq_P1.json')
```
`plot_q1asm_files` does the same for multiple files.
```Python
    from q1simulator import plot_q1asm_files, PlotDef

    path = 'demo_data'
    plot_q1asm_files([
            PlotDef(path + r'/q1seq_P1.json', 'P1', [0]),
            PlotDef(path + r'/q1seq_P2.json', 'P2', [1]),
            PlotDef(path + r'/q1seq_q1.json', 'q1', [2,3], lo_frequency=20e6),
            PlotDef(path + r'/q1seq_R1.json', 'R1', []),
            ])
```
See demo directory for some examples.

The viewer can be executed from the commandline to view a single sequence file:
`python -m q1simulator.q1viewer q1simulator\demo\demo_data\q1seq_P1.json`

# Simulator rendering limits
The maximum length of the rendered output is limited to 2 ms,
because plots with more points do not perform well.
The maximum number of execution cycles is limited to 1e7,
because 1 cycle take 1 to 6 us in the simulator. So, after
10 to 60 seconds the simulator aborts execution of a sequencer.

The limits can be changed:
```Python
    sim.config('max_render_time', 10_000_000)
    sim.config('max_core_cycles', 1e9)
```

# Implemented methods
Q1Simulator checks the installed version of qblox_instruments and mimics the
API of that version.

The following QCM/QRM methods and parameters are implemented by
the simulator and mimic behavior of the cluster / module / sequencer:
- module_type
- is_qcm_type, is_qrm_type, is_rf_type
- reset
- get_system_state
- get_num_system_error
- get_system_error
- get_sequencer_state
- arm_sequencer
- start_sequencer
- stop_sequencer
- get_acquisition_state
- get_acquisitions
- delete_acquisition_data
- update_sequence
- connect_sequencer
- disconnect_inputs
- disconnect_outputs

Simulated sequencer qcodes parameters:
- gain_awg_pathX
- offset_awg_pathX
- nco_freq
- mod_en_awg
- demod_en_acq
- connect_outX
- sequence
- mixer_corr_gain_ratio
- mixer_corr_phase_offset_degree
- triggerXX_count_threshold
- triggerXX_threshold_invert
- integration_length_acq
- thresholded_acq_rotation
- thresholded_acq_threshold
- thresholded_acq_trigger_en
- thresholded_acq_trigger_address
- thresholded_acq_trigger_invert
- ttl_acq_auto_bin_incr_en

All other methods and parameters only write the passed value to the logger.

# Setting simulator acquisition data
Acquisition mock data can be set with `set_acquisition_mock_data`.
The data should passed in a list for multiple runs of the sequence.
For every run there should be a list with entries for every `acquire`
call. The entry for an acquire call is used for both paths.
If it is a single float value then it is used for both paths.
If it is a complex value then the real part is used for path 0 and
the imaginary part for path 1.
If it is a sequence of two floats then the first is used for path 0
and the second for path 1.

When `repeat=True` the list with data is repeated, otherwise
an exception is raised when the sequencer is started when the
list with mock data is exhausted.

Passing None for mock data resets the default behavior of the sequencer.
The default behavior is to return the real-time timestamp of the
acquire call.

Example:
```Python
    # set data for 1 run to return the values 0 till 19 on path 0 and path 1
    data = [np.arange(20)]
    sim.sequencers[0].set_acquisition_mock_data(data)

    # set data for every run to return IQ values with changing phase
    # on path 0 and 1
    data = [np.exp(np.pi*1j*np.arange(20)/10)]
    sim.sequencers[0].set_acquisition_mock_data(data, repeat=True)

    # set data for every run to return the values 0 till 19 on path 0
    # and 100 till 119 on path 1.
    data = [np.arange(20) + 1j*np.arange(100,120)]
    sim.sequencers[0].set_acquisition_mock_data(data, repeat=True)

    # set data for 2 runs to return the values 0 till 19 on the first run
    # and 100 till 119 on the second run.
    data2 = [np.arange(20), np.arange(100, 120)]
    sim.sequencers[0].set_acquisition_mock_data(data2)

    # Return 0...19 and 100...119 alternatingly.
    sim.sequencers[0].set_acquisition_mock_data(data2, repeat=True)

    # reset default behaviour.
    sim.sequencers[0].set_acquisition_mock_data(None)
```

# Conditional execution
The simulator generates triggers according to the acquisition thresholds.
At startup the simulator determines the dependencies between trigger producers
and trigger consumers. It executes the trigger producing sequencers
before the trigger consuming sequencers.

The simulator takes the trigger network latency into account, but does not
check whether triggers overlap.

The configuration setting `acq_trigger_value` allows to evaluate the sequence
for a specified trigger value. `sim.config('acq_trigger_value', 1)` sends a trigger
after every acquisition. `sim.config('acq_trigger_value', 0)` sends no triggers.
If `acq_trigger_value` is None then the simulator uses the threshold on the
acquisition data.

The condition evaluation can also be forced with the property `value_for_condition`
of the module. If this value is 1 then OR, AND and XOR evaluate to true.
If this value is 0 then NOR, NAND and XNOR evaluate to true.
The feature is suitable for simple test.

# Simulator output
The simulator has some methods to show the simulator output.
- `plot()` shows pyplot charts with the rendered output.
- `get_output()` returns the rendered output in dictionary with numpy arrays.
- `config("skip_loops", ("_start", ))` stops rendering when Q1Pulse main sequence is executed once.
- `print_acquisitions()` prints the path0 and path1 data and average counts.
- `print_registers()` prints the contents of the Q1 registers.

# Simulator logging
Q1Simulator has a logging feature to help with code debugging.
It uses of special comment line in the Q1ASM code.
The comment line should start with `#Q1Sim:` and is followed by the
simulator log command with the format log "message",register,options.
The options are:
* R: log register value
* T: log q1 and real-time executer time.
* F: format register value as Q1Pulse float

Example:
```
    #Q1Sim: log "t_wait",R3,TR
          wait  R3
    #Q1Sim: log "after",none,T
```
Output:
```
    t_wait:          12 (0000000C) q1:   324 rt:   908 ns
    after:  q1:   332 rt:   920 ns
```

# About Q1Simulator
One day after I had been working on the generation of Q1ASM I thought
it would be fun to write a interpreter to execute the generated Q1ASM.
A few hours later I had a parser, an interpreter and an output renderer
plotting the output of the Q1ASM file that I had generated earlier
that day. Above all it showed there was an error in the timing of
generated signal. The next day I started to use Q1Simulator to
check the code I generated with Q1Pulse and pulselib. I found
several bugs in the generated code and in Q1Simulator.

Q1Simulator evolved into a useful tool.
Now it is available to all.

*Sander de Snoo*

