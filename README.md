# Q1Simulator
Simulator to execute Q1ASM and render the output channels.
The simulator can be used like a `Pulsar` or `Cluster` object
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

# Example

    from q1simulator import Q1Simulator as Pulsar

    sim = Pulsar('q1sim', sim_type='QCM')
    sim.sequencer0.sequence('my_sequence.json')
    sim.arm_sequencer(0)
    sim.start_sequencer()
    sim.get_sequencer_state(0)
    sim.plot()
    sim.print_acquisitions()

# Cluster

A simulated Cluster can be created with the modules
specified in a dictionary with  slot number and module type.

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


# Q1ASM file viewer
`plot_q1asm_file` is a simple function that reads a file
and plots and prints the results.
    from q1simulator import plot_q1asm_file

    path = 'demo_data'
    plot_q1asm_file(path + r'/q1seq_P1.json')

`plot_q1asm_files` does the same for multiple files.
    from q1simulator import plot_q1asm_files, PlotDef

    path = 'demo_data'
    plot_q1asm_files([
            PlotDef(path + r'/q1seq_P1.json', 'P1', [0]),
            PlotDef(path + r'/q1seq_P2.json', 'P2', [1]),
            PlotDef(path + r'/q1seq_q1.json', 'q1', [2,3], lo_frequency=20e6),
            PlotDef(path + r'/q1seq_R1.json', 'R1', []),
            ])

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
    sim.config('max_render_time', 10_000_000)
    sim.config('max_core_cycles', 1e9)


# Implemented methods
The following QCM/QRM methods and parameters are implemented by
the simulator and mimic behavior of the device:
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
- start_adc_calib
- sequencer0.nco_freq
- sequencer0.mod_en_awg
- sequencer0.demod_en_acq
- sequencer0.channel_map_path0_out0_en
- sequencer0.channel_map_path1_out1_en
- sequencer0.channel_map_path0_out2_en
- sequencer0.channel_map_path1_out3_en
- sequencer0.sequence
- sequencer0.mixer_corr_gain_ratio
- sequencer0.mixer_corr_phase_offset_degree

All other parameters only print the passed value.

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

# Conditional execution
The simulator generates triggers according to the acquisition thresholds. 
At startup the simulator determines the dependencies between trigger producers 
and trigger consumers. It executes the trigger producing sequencers
before the trigger consuming sequencers.

The simulator takes the trigger network latency into account, but does not
check whether triggers overlap. 


# Simulator output
The simulator has some methods to show the simulator output.
- `plot()` shows pyplot charts with the rendered output.
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

    #Q1Sim: log "t_wait",R3,TR
          wait  R3
    #Q1Sim: log "after",none,T

Output:

    t_wait:          12 (0000000C) q1:   324 rt:   908 ns
    after:  q1:   332 rt:   920 ns


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

